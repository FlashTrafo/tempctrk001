#!/usr/bin/env python

#SBATCH --job-name=cth05

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uivan@student.kit.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import sys
sys.path.append("/pfs/data5/home/kit/tm/px6680/cth2/ctrknas/sfinver3/")

import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from scipy.special import softmax
import argparse
import os
import random

# from general_functions.dataloaders import get_loaders, get_test_loader
from general_functions.utils import get_logger, weights_init, load, create_directories_from_list, \
                                    check_tensor_in_list, writh_new_ARCH_to_fbnet_modeldef
from supernet_functions.lookup_table_builder import LookUpTable
# from supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
from supernet_functions.model_single import FBNet_Stochastic_SuperNet, SupernetLoss
# from supernet_functions.training_functions_supernet import TrainerSupernet
from supernet_functions.training_single import TrainerSupernet
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
# from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH
from general_functions.dataset_factory import get_dataset
from general_functions.opts import optss

    
parser = argparse.ArgumentParser("action")

parser.add_argument('--train_or_sample', type=str, default='train', \
                    help='train means training of the SuperNet, sample means sample from SuperNet\'s results')
parser.add_argument('--architecture_name', type=str, default='testnet004', \
                    help='Name of an architecture to be sampled')
parser.add_argument('--hardsampling_bool_value', type=str, default='True', \
                    help='If not False or 0 -> do hardsampling, else - softmax sampling')
#######################
parser.add_argument('--pathtb', type=str, default='./supernet_functions/logs0602/tb/', \
                    help='path to tensorboard')
parser.add_argument('--pathlog', type=str, default='./supernet_functions/logs0602/logger', \
                    help='path to logger')
parser.add_argument('--pathsave', type=str, default='./supernet_functions/logs0602/cur_model.pth', \
                    help='current model save path')
parser.add_argument('--pathbest', type=str, default='./supernet_functions/logs0602/best_model.pth', \
                    help='best model save path')
parser.add_argument('--sparse', type=float, default=0.00005, \
                    help='sparse rate')
parser.add_argument('--pathtrain', type=str, default='./supernet_functions/logs0602/fit_model.pth', \
                    help='limit overfitting model save path')
parser.add_argument('--savefit', action='store_true', default=False)
parser.add_argument('--wlr', type=float, default=0.0005)
parser.add_argument('--tlr', type=float, default=0.0005)

args = parser.parse_args()

def train_supernet():
    manual_seed = 1
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    # create_directories_from_list([CONFIG_SUPERNET['logging']['path_to_tensorboard_logs']]) ############
    create_directories_from_list(args.pathtb)
    
    # logger = get_logger(CONFIG_SUPERNET['logging']['path_to_log_file']) #####################
    logger = get_logger(args.pathlog)
    # writer = SummaryWriter(log_dir=CONFIG_SUPERNET['logging']['path_to_tensorboard_logs']) ############
    writer = SummaryWriter(args.pathtb)
    
    #### LookUp table consists all information about layers
    lookup_table = LookUpTable(calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'])
    
    #### DataLoading
    opt = optss()
    Dataset = get_dataset(opt.dataset)
    # opt = optss().update_dataset_info_and_set_heads(opt, Dataset)
    opt.update_dataset_info_and_set_heads(Dataset)
    train_w_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True#, drop_last=True
    )
    train_thetas_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'grad'), batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True#, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=16,
        pin_memory=True)
    # train_w_loader, train_thetas_loader = get_loaders(CONFIG_SUPERNET['dataloading']['w_share_in_train'],
    #                                                   CONFIG_SUPERNET['dataloading']['batch_size'],
    #                                                   CONFIG_SUPERNET['dataloading']['path_to_save_data'],
    #                                                   logger)
    # test_loader = get_test_loader(CONFIG_SUPERNET['dataloading']['batch_size'],
    #                               CONFIG_SUPERNET['dataloading']['path_to_save_data'])
    
    #### Model
    model = FBNet_Stochastic_SuperNet(lookup_table).cuda()
    model = model.apply(weights_init)
    # model.init_arch_params()  #########################################
    # model = nn.DataParallel(model, device_ids=[0])
    
    #### Loss, Optimizer and Scheduler
    criterion = SupernetLoss(opt).cuda()

    # thetas_params = [param for name, param in model.named_parameters() if 'thetas' in name]
    # thetas_params = model.architecture_parameters()  ##############################################
    # params_except_thetas = [param for param in model.parameters() if not check_tensor_in_list(param, thetas_params)]
    weights_params = model.weight_parameters()

    # w_optimizer = torch.optim.SGD(params=weights_params,
    #                               lr=CONFIG_SUPERNET['optimizer']['w_lr'], 
    #                               momentum=CONFIG_SUPERNET['optimizer']['w_momentum'],
    #                               weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
    
    w_optimizer = torch.optim.SGD(params=weights_params,
                                  lr=args.wlr, 
                                  momentum=CONFIG_SUPERNET['optimizer']['w_momentum'],
                                  weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
    
    # theta_optimizer = torch.optim.Adam(params=thetas_params,
    #                                    lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
    #                                    weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])
    
    # theta_optimizer = torch.optim.Adam(params=thetas_params,
    #                                lr=args.tlr,
    #                                weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])  ##############
    #############################################################################################################

    last_epoch = -1
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer,
                                                             T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
                                                             last_epoch=last_epoch)

    # if resume:
    #     checkpoint = torch.load(resume)
    #     model.load_state_dict(checkpoint["state_dict"])
    #     w_optimizer.load_state_dict(checkpoint["w_optimizer"])
    #     theta_optimizer.load_state_dict(checkpoint["theta_optimizer"])
    #     w_scheduler = checkpoint["w_scheduler"]

    #### Training Loop
    # trainer = TrainerSupernet(criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer, args)   ###################
    trainer = TrainerSupernet(criterion, w_optimizer, w_scheduler, logger, writer, args)
    # trainer.train_loop(train_w_loader, train_thetas_loader, test_loader, model) #####################
    trainer.train_loop(train_w_loader, test_loader, model)

# Arguments:
# hardsampling=True means get operations with the largest weights
#             =False means apply softmax to weights and sample from the distribution
# unique_name_of_arch - name of architecture. will be written into fbnet_building_blocks/fbnet_modeldef.py
#                       and can be used in the training by train_architecture_main_file.py
# def sample_architecture_from_the_supernet(unique_name_of_arch, hardsampling=True):
#     logger = get_logger(CONFIG_SUPERNET['logging']['path_to_log_file'])
#
#     lookup_table = LookUpTable()
#     model = FBNet_Stochastic_SuperNet(lookup_table, cnt_classes=10).cuda()
#     model = nn.DataParallel(model)
#
#     load(model, CONFIG_SUPERNET['train_settings']['path_to_save_model'])
#
#     ops_names = [op_name for op_name in lookup_table.lookup_table_operations]
#     neck_names = [op_name for op_name in lookup_table.lookup_table_operations_fpn] #######
#     head_names = [op_name for op_name in lookup_table.lookup_table_operations_head] #########
#     cnt_ops = len(ops_names)
#
#     arch_operations=[]
#     if hardsampling:
#         for layer in model.module.stages_to_search:
#             arch_operations.append(ops_names[np.argmax(layer.thetas.detach().cpu().numpy())])
#         for layer in model.module.neck: ##############
#             arch_operations.append(neck_names[np.argmax(layer.thetas.detach().cpu().numpy())])
#         for layer in model.module.head: ###############
#             arch_operations.append(head_names[np.argmax(layer.thetas.detach().cpu().numpy())])
#     else:
#         rng = np.linspace(0, cnt_ops - 1, cnt_ops, dtype=int)
#         for layer in model.module.stages_to_search:
#             distribution = softmax(layer.thetas.detach().cpu().numpy())
#             arch_operations.append(ops_names[np.random.choice(rng, p=distribution)])
#         for layer in model.module.neck:
#             distribution = softmax(layer.thetas.detach().cpu().numpy())
#             arch_operations.append(neck_names[np.random.choice(rng, p=distribution)])
#         for layer in model.module.head:
#             distribution = softmax(layer.thetas.detach().cpu().numpy())
#             arch_operations.append(head_names[np.random.choice(rng, p=distribution)])
#
#     logger.info("Sampled Architecture: " + " - ".join(arch_operations))
#     writh_new_ARCH_to_fbnet_modeldef(arch_operations, my_unique_name_for_ARCH=unique_name_of_arch)
#     logger.info("CONGRATULATIONS! New architecture " + unique_name_of_arch \
#                 + " was written into fbnet_building_blocks/fbnet_modeldef.py")


def sample_architecture_from_the_supernet(unique_name_of_arch, hardsampling=True):
    logger = get_logger(CONFIG_SUPERNET['logging']['path_to_log_file'])  #############################

    lookup_table = LookUpTable()
    model = FBNet_Stochastic_SuperNet(lookup_table).cuda()
    # model = nn.DataParallel(model)

    load(model, CONFIG_SUPERNET['train_settings']['path_to_save_model'])

    ops_names = [op_name for op_name in lookup_table.lookup_table_operations]
    neck_names = [op_name for op_name in lookup_table.lookup_neck_operations]  #######
    head_names = [op_name for op_name in lookup_table.lookup_head_operations]  #########
    cnt_ops = len(ops_names)

    # arch_operations = []
    sampled = []
    sampled_idx = []
    # for layer in model.module.stages_to_search: # .module??????????????
    #     sampled_idx.append(np.argmax(layer.AP_path_alpha.detach().cpu().numpy()))
    #     sampled.append(ops_names[np.argmax(layer.AP_path_alpha.detach().cpu().numpy())])
    # sampled_idx.append(0)
    # sampled.append('no')
    # for layer in model.module.neck:  ##############
    #     sampled_idx.append(np.argmax(layer.AP_path_alpha.detach().cpu().numpy()))
    #     sampled.append(ops_names[np.argmax(layer.AP_path_alpha.detach().cpu().numpy())])
    # sampled_idx.append(0)
    # sampled.append('no')
    # for layer in model.module.head:  ###############
    #     sampled_idx.append(np.argmax(layer.AP_path_alpha.detach().cpu().numpy()))
    #     sampled.append(ops_names[np.argmax(layer.AP_path_alpha.detach().cpu().numpy())])
    
    for layer in model.stages_to_search: # .module??????????????
        sampled_idx.append(np.argmax(layer.AP_path_alpha.detach().cpu().numpy()))
        sampled.append(ops_names[np.argmax(layer.AP_path_alpha.detach().cpu().numpy())])
    sampled_idx.append(0)
    sampled.append('no')
    for layer in model.neck:  ##############
        sampled_idx.append(np.argmax(layer.AP_path_alpha.detach().cpu().numpy()))
        sampled.append(neck_names[np.argmax(layer.AP_path_alpha.detach().cpu().numpy())])
    sampled_idx.append(0)
    sampled.append('no')
    for layer in model.head:  ###############
        sampled_idx.append(np.argmax(layer.AP_path_alpha.detach().cpu().numpy()))
        sampled.append(head_names[np.argmax(layer.AP_path_alpha.detach().cpu().numpy())])

    stat_dict = model.state_dict()
    print(sampled)
    print(sampled_idx)
    for a, b in stat_dict.items():
        if 'out_layer' in a:
            print(a)
            # print(stat_dict[a].shape)
        # if 'head' not in a and 'stage' not in a and 'neck' not in a:
            # print(a,': ',b.shape)
    
 
    return sampled, sampled_idx, stat_dict
    # logger.info("Sampled Architecture: " + " - ".join(arch_operations))
    # writh_new_ARCH_to_fbnet_modeldef(arch_operations, my_unique_name_for_ARCH=unique_name_of_arch)
    # logger.info("CONGRATULATIONS! New architecture " + unique_name_of_arch \
    #             + " was written into fbnet_building_blocks/fbnet_modeldef.py")


if __name__ == "__main__":
    assert args.train_or_sample in ['train', 'sample']
    if args.train_or_sample == 'train':
        train_supernet()
    # elif args.train_or_sample == 'sample':
    #     # assert args.architecture_name != '' and args.architecture_name not in MODEL_ARCH
    #     hardsampling = False if args.hardsampling_bool_value in ['False', '0'] else True
    #     sampled, sampled_idx, stat_dict = sample_architecture_from_the_supernet(unique_name_of_arch=args.architecture_name, hardsampling=hardsampling)

#sn01 0.005 0.002
#sn02 0.0005 0.002 failed
#sn03 0.0005 0.005 f
#sn04 0.005 0.005 f
#sn02 0.0005 0.001
#sn03 0.005 0.001
#s02 ???
#sn03 0.0005 0.002  save in sn01
#sn02 0.005 0.002

#0201 0.0005 0.0002
#0202 0.005 0.0002
#0203 0.0001 0.0002
#0204 0.00005 0.0002
#0205 0.0005 0.002
#0206 0.005 0.002
#0207 0.0001 0.002
#0208 0.00005 0.002

