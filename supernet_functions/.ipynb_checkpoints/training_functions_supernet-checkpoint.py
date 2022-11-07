import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from general_functions.utils import AverageMeter, save, accuracy
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from fbnet_building_blocks.fbnet_builder import ConvBNRelu
from supernet_functions.model_supernet import MixedOperation

class BNOptimizer:
    @staticmethod
    def updateBN(model, s):
        # stp = 0
        # nstp = 0
        for mm in model.children():
            if isinstance(mm, nn.ModuleList):
                for m in mm.modules():
                    if isinstance(m, ConvBNRelu):
                        bn_module = m[1]
                        # if bn_module.weight.grad is None:
                            # stp += 1
                        # else:
                        if bn_module.weight.grad is not None:
                            bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1
                            # nstp += 1
                # print("stp:!!!! ", stp)
                # print("nstp:   ", nstp)
        # print("stp:!!!! ", stp)
        # print("nstp:   ", nstp)


class TrainerSupernet:
    # def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer):
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer, args):
        self.top1       = AverageMeter()
        self.top3       = AverageMeter()
        self.losses     = AverageMeter()
        self.losses_lat = AverageMeter()
        self.losses_ce  = AverageMeter()
        ###
        self.losses_hm  = AverageMeter()
        self.losses_wh  = AverageMeter()
        self.losses_reg = AverageMeter()
        self.losses_trk = AverageMeter()
        
        
        self.logger = logger
        self.writer = writer
        
        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler
        
        self.temperature                 = CONFIG_SUPERNET['train_settings']['init_temperature']
        self.exp_anneal_rate             = CONFIG_SUPERNET['train_settings']['exp_anneal_rate'] # apply it every epoch
        self.cnt_epochs                  = CONFIG_SUPERNET['train_settings']['cnt_epochs']
        self.train_thetas_from_the_epoch = CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch']
        self.print_freq                  = CONFIG_SUPERNET['train_settings']['print_freq']
        # self.path_to_save_model          = CONFIG_SUPERNET['train_settings']['path_to_save_model'] ###
        # self.scale_sparse_rate           = CONFIG_SUPERNET['train_settings']['scale_sparse_rate'] ###
        # self.warm_up_epochs              = CONFIG_SUPERNET['train_settings']['warm_up_epochs']
        self.path_to_save_model          = args.pathsave
        self.scale_sparse_rate           = args.sparse
        self.path_to_best_model          = args.pathbest
        self.path_to_cut_model           = args.pathtrain
        self.savefit                     = args.savefit
    
    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):
        
        # best_top1 = 0.0
        least_train_loss = 65535
        best_loss = 65535
        
        # firstly, train weights only
        for epoch in range(self.train_thetas_from_the_epoch):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Firstly, start to train weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            self.w_scheduler.step()
        
        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('learning_rate/theta', self.theta_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Start to train weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            self.w_scheduler.step()
            
            self.logger.info("Start to train theta for epoch %d" % (epoch))
            self._training_step(model, train_thetas_loader, self.theta_optimizer, epoch, info_for_logger="_theta_step_")
            
            loss_epoch_tr = self.losses.get_avg()
            if self.savefit:
                if loss_epoch_tr < least_train_loss:
                    least_train_loss = loss_epoch_tr
                    print('Model with lowest train loss {}. Save model'.format(loss_epoch_tr))
                    save(model, self.path_to_cut_model)
                    
            
            # top1_avg = self._validate(model, test_loader, epoch)
            self._validate(model, test_loader, epoch)
            # if best_top1 < top1_avg:
            #     best_top1 = top1_avg
            #     self.logger.info("Best top1 acc by now. Save model")
            save(model, self.path_to_save_model)
            loss_epoch = self.losses.get_avg()
            if loss_epoch < best_loss:
                best_loss = loss_epoch
                print('Model with lowest val loss {}. Save model'.format(loss_epoch))
                save(model, self.path_to_best_model)
            
            self.temperature = self.temperature * self.exp_anneal_rate
       
    def _training_step(self, model, loader, optimizer, epoch, info_for_logger=""): #,arch_mode=None
        model = model.train()
        start_time = time.time()
        
        if info_for_logger == "_theta_step_":
            MixedOperation.MODE = 'grad'
                
        for step, X in enumerate(loader):
            # X = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # N = X.shape[0]
            N = X['image'].shape[0]
            for a in X.keys():
                X[a] = X[a].cuda()

            latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
            model.reset_binary_gates()
            model.unused_modules_off()
            outs, latency_to_accumulate = model(X, latency_to_accumulate) # go to model_sup
            loss, track_loss = self.criterion(outs, X, latency_to_accumulate, self.losses_ce, self.losses_lat, N)  # y = target (label)
            optimizer.zero_grad()
            loss.backward()
            
            if info_for_logger == "_w_step_":
                BNOptimizer.updateBN(model, self.scale_sparse_rate)
            if info_for_logger == "_theta_step_":
                model.set_arch_param_grad()
                print(loss)
            optimizer.step()
            if info_for_logger == "_theta_step_":
                model.rescale_updated_arch_param()
            model.unused_modules_back()
            self._intermediate_stats_logging(outs, track_loss, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")
               
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train='train')
        # for avg in [self.top1, self.top3, self.losses]:
        for avg in [self.losses, self.losses_ce, 
                    self.losses_hm, self.losses_wh, self.losses_reg, self.losses_trk]:
            avg.reset()
            
        if info_for_logger == "_theta_step_":
            MixedOperation.MODE = None
        
    def _validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            # for step, (X, y) in enumerate(loader):
            for step, X in enumerate(loader):
                # X, y = X.cuda(), y.cuda()
                N = X['image'].shape[0]
                # for a in X:
                # for a, b in X.items():
                    # b = b.cuda()  # ?????????????????
                # X = X.cuda()
                for a in X.keys():
                    X[a] = X[a].cuda()
                    
                model.set_chosen_op_active()
                model.unused_modules_off()
                    
                latency_to_accumulate = torch.Tensor([[0.0]]).cuda()
                outs, latency_to_accumulate = model(X, latency_to_accumulate)
                loss, track_loss = self.criterion(outs, X, latency_to_accumulate, self.losses_ce, self.losses_lat, N)
                model.unused_modules_back()

                self._intermediate_stats_logging(outs, track_loss, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")
                
        # top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='val')
        # for avg in [self.top1, self.top3, self.losses]:
        for avg in [self.losses, self.losses_ce, 
                    self.losses_hm, self.losses_wh, self.losses_reg, self.losses_trk]:
            avg.reset()
        # return top1_avg
    
    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=''):
        # self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss'+info_for_logger, self.losses.get_avg(), epoch)
        # self.writer.add_scalar('train_vs_val/'+val_or_train+'_top1'+info_for_logger, self.top1.get_avg(), epoch)
        # self.writer.add_scalar('train_vs_val/'+val_or_train+'_top3'+info_for_logger, self.top3.get_avg(), epoch)
        # self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_lat'+info_for_logger, self.losses_lat.get_avg(), epoch)
        # self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_ce'+info_for_logger, self.losses_ce.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss'+info_for_logger, self.losses.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_lat'+info_for_logger, self.losses_lat.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss_trk_tot'+info_for_logger, self.losses_ce.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss_hm'+info_for_logger, self.losses_hm.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss_reg'+info_for_logger, self.losses_reg.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss_wh'+info_for_logger, self.losses_wh.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss_trk'+info_for_logger, self.losses_trk.get_avg(), epoch)
        
        top1_avg = self.top1.get_avg()
        self.logger.info(info_for_logger+val_or_train + ": [{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(
            epoch+1, self.cnt_epochs, top1_avg, time.time() - start_time)) #XXXXXXXXXXXXXXXXXXXX
        
    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        # prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        # self.top1.update(prec1.item(), N)
        # self.top3.update(prec3.item(), N)
        ########
        # self.writer.add_scalar('hm_loss', y['hm'], epoch)
        # self.writer.add_scalar('reg_loss', y['reg'], epoch)
        # self.writer.add_scalar('wh_loss', y['wh'], epoch)
        # self.writer.add_scalar('tracking_loss', y['tracking'], epoch)
        ########
        self.losses_hm.update(y['hm'], N)
        self.losses_wh.update(y['wh'], N)
        self.losses_reg.update(y['reg'], N)
        self.losses_trk.update(y['tracking'], N)
        
        
#         if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
#             self.logger.info(val_or_train+
#                ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
#                "Prec@(1,3) ({:.1%}, {:.1%}), ce_loss {:.3f}, lat_loss {:.3f}".format(
#                    epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
#                    self.top1.get_avg(), self.top3.get_avg(), self.losses_ce.get_avg(), self.losses_lat.get_avg()))
         
    
    
    #########################################################################################
        # if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
        #     self.logger.info(val_or_train+
        #        ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
        #        ", total_loss {:.3f}, lat_loss {:.3f}, hm_loss {:.3f}, reg_loss {:.3f}, "
        #        "wh_loss {:.3f}, offset_loss {:.3f}".format(
        #            epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
        #            self.losses_ce.get_avg(), self.losses_lat.get_avg(), 
        #            y['hm'], y['reg'], y['wh'], y['tracking']))
    ########################################################################################
        if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(val_or_train+
               ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
               ", total_loss {:.3f}, lat_loss {:.3f}, hm_loss {:.3f}, reg_loss {:.3f}, "
               "wh_loss {:.3f}, offset_loss {:.3f}".format(
                   epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
                   self.losses_ce.get_avg(), self.losses_lat.get_avg(), 
                   self.losses_hm.get_avg(), self.losses_reg.get_avg(),
                   self.losses_wh.get_avg(), self.losses_trk.get_avg()))
    
    
#     def _valid_logging(self, start_time, epoch, metrics_output):
#     precision, recall, AP, f1, ap_class = metrics_output

#         self.writer.add_scalar('valid_precision', precision.mean().item(), epoch)
#         self.writer.add_scalar('valid_recall', recall.mean().item(), epoch)
#         self.writer.add_scalar('valid_mAP', AP.mean().item(), epoch)
#         self.writer.add_scalar('valid_f1', f1.mean().item(), epoch)

#         self.logger.info("valid : [{:3d}/{}] Final Precision {:.4%}, Time {:.2f}".format(
#             epoch + 1, self.cnt_epochs, AP.mean().item(), time.time() - start_time))

#     def _train_logging(self, loss, ce, lat, loss_components, step, epoch,
#                                     len_loader, info_for_logger=''):

#         self.writer.add_scalar('total_loss', loss.item(), epoch)
#         self.writer.add_scalar('ce_loss', ce.item(), epoch)
#         self.writer.add_scalar('latency_loss', lat.item(), epoch)
#         self.writer.add_scalar('iou_loss', loss_components[0].item(), epoch)
#         self.writer.add_scalar('obj_loss', loss_components[1].item(), epoch)
#         self.writer.add_scalar('cls_loss', loss_components[2].item(), epoch)

#         if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
#             self.logger.info("training" + info_for_logger +
#                              ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
#                              "ce_loss {:.3f}, lat_loss {:.3f} "
#                              "iou_loss {:.3f}, obj_loss {:.3f}, cls_loss {:.3f}".format(
#                                  epoch + 1, self.cnt_epochs, step, len_loader - 1, loss.item(), ce.item(), lat.item(),
#                                  loss_components[0].item(), loss_components[1].item(), loss_components[2].item()))