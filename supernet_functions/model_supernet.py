import torch
import math
from torch import nn
from collections import OrderedDict
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Upsample
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from general_functions.trackloss.trackgenericloss import GenericLoss
# from general_functions.trackloss.trackselfloss import GenericLoss
from general_functions.opts import optss
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


# def detach_variable(inputs):
#     if isinstance(inputs, tuple):
#         return tuple([detach_variable(x) for x in inputs])
#     else:
#         x = inputs.detach()
#         x.requires_grad = inputs.requires_grad
#         #######################################
#         x.requires_grad = False
#         #######################################
#         return x


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        # detached_x = detach_variable(x)
        detached_x = x
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors
        # detached_x.requires_grad = True
        # output.requires_grad = True
        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None




class MixedOperation(nn.Module):
    MODE = None
    # Arguments:
    # proposed_operations is a dictionary {operation_name : op_constructor}
    # latency is a dictionary {operation_name : latency}
    def __init__(self, layer_parameters, proposed_operations, latency):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.latency = [latency[op_name] for op_name in ops_names]
        # self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
        # self.n_choices = len(proposed_operations) ######
        self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))
        self.log_prob = None
        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))
        self._unused_modules_backbone = []
        self._unused_modules_neck = []
        self._unused_modules_head = []
        self.active_index = [0]
        self.inactive_index = None
        self.current_prob_over_ops = None

    @property
    def n_choices(self):
        return len(self.ops)

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.ops[index]

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.ops[self.active_index[0]]

    @property
    def random_op(self):
        index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
        return self.ops[index]

    # def is_zero_layer(self):
    #     return self.active_op.is_zero_layer()

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def forward(self, x, latency_to_accumulate):
        # output = 0
        # for _i in self.active_index:
        #     oi = self.ops[_i](x)
        #     latency = self.latency[_i]
        #     output = output + self.AP_path_wb[_i] * oi
        #     latency_to_accumulate = latency_to_accumulate + self.AP_path_alpha[_i] * latency
        # for _i in self.inactive_index:
        #     oi = self.ops[_i](x)
        #     latency = self.latency[_i]
        #     output = output + self.AP_path_wb[_i] * oi.detach()
        #     # latency_to_accumulate = latency_to_accumulate + self.AP_path_alpha[_i] * latency.detach()
        #     latency_to_accumulate = latency_to_accumulate + self.AP_path_alpha[_i] * latency
        if MixedOperation.MODE == 'grad':
            def run_function(candidate_ops, active_id):
                def forward(_x):
                    return candidate_ops[active_id](_x)

                return forward

            def backward_function(candidate_ops, active_id, inactive_id, binary_gates):
                def backward(_x, _output, grad_output):
                    binary_grads = torch.zeros_like(binary_gates.data)
                    with torch.no_grad():
                        inactive_out_k = candidate_ops[inactive_id](_x.data)
                        active_out_k = _output.data
                        binary_grads[inactive_id] = torch.sum(inactive_out_k * grad_output)
                        binary_grads[active_id] = torch.sum(active_out_k * grad_output)
                    return binary_grads

                return backward

            output = ArchGradientFunction.apply(
                x, self.AP_path_wb, run_function(self.ops, self.active_index[0]),
                backward_function(self.ops, self.active_index[0], self.inactive_index[0], self.AP_path_wb)
            )
            for _i in range(self.n_choices):
                if _i in self.active_index:
                    latency_to_accumulate = latency_to_accumulate + self.latency[_i] * self.probs_over_ops[_i]
                else:
                    latency_to_accumulate = latency_to_accumulate + self.latency[_i] * self.probs_over_ops[_i]

        else:
            output = self.active_op(x)
            for _i in range(self.n_choices):
                if _i in self.active_index:
                    latency_to_accumulate = latency_to_accumulate + self.latency[_i] * self.probs_over_ops[_i]
                else:
                    latency_to_accumulate = latency_to_accumulate + self.latency[_i] * self.probs_over_ops[_i]  ###### ??????

        return output, latency_to_accumulate

    # def binarize(self): ######
    #     """ prepare: active_index, inactive_index, AP_path_wb, log_prob (optional), current_prob_over_ops (optional) """
    #     self.log_prob = None
    #     # reset binary gates
    #     self.AP_path_wb.data.zero_()
    #     # binarize according to probs
    #     probs = self.probs_over_ops
    #     # if MixedEdge.MODE == 'two':
    #         # sample two ops according to `probs`
    #     sample_op = torch.multinomial(probs.data, 2, replacement=False)
    #     probs_slice = F.softmax(torch.stack([
    #         self.AP_path_alpha[idx] for idx in sample_op
    #     ]), dim=0)
    #     self.current_prob_over_ops = torch.zeros_like(probs)
    #     for i, idx in enumerate(sample_op):
    #         self.current_prob_over_ops[idx] = probs_slice[i]
    #     # chose one to be active and the other to be inactive according to probs_slice
    #     c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
    #     active_op = sample_op[c].item()
    #     inactive_op = sample_op[1 - c].item()
    #     self.active_index = [active_op]
    #     self.inactive_index = [inactive_op]
    #     # set binary gate
    #     self.AP_path_wb.data[active_op] = 1.0
    #     # else:
    #     #     sample = torch.multinomial(probs.data, 1)[0].item()
    #     #     self.active_index = [sample]
    #     #     self.inactive_index = [_i for _i in range(0, sample)] + \
    #     #                           [_i for _i in range(sample + 1, self.n_choices)]
    #     #     self.log_prob = torch.log(probs[sample])
    #     #     self.current_prob_over_ops = probs
    #     #     # set binary gate
    #     #     self.AP_path_wb.data[sample] = 1.0
    #     # avoid over-regularization
    #     for _i in range(self.n_choices):
    #         for name, param in self.ops[_i].named_parameters():
    #             param.grad = None
    
    def binarize(self):
        """ prepare: active_index, inactive_index, AP_path_wb, log_prob (optional), current_prob_over_ops (optional) """
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_() # [0,0,0,0,0]
        # binarize according to probs
        probs = self.probs_over_ops 
        if MixedOperation.MODE == 'grad':
            # sample two ops according to `probs`
            sample_op = torch.multinomial(probs.data, 2, replacement=False) 
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in sample_op
            ]), dim=0)
            self.current_prob_over_ops = torch.zeros_like(probs)
            for i, idx in enumerate(sample_op):
                self.current_prob_over_ops[idx] = probs_slice[i] 
            # chose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
            active_op = sample_op[c].item() 
            inactive_op = sample_op[1 - c].item()
            self.active_index = [active_op] # [3]
            self.inactive_index = [inactive_op] # [2]
            # set binary gate
            self.AP_path_wb.data[active_op] = 1.0 # [0,0,0,1,0]
        else:
            sample = torch.multinomial(probs.data, 1)[0].item() 
            self.active_index = [sample] # [2]
            self.inactive_index = [_i for _i in range(0, sample)] + \
                                  [_i for _i in range(sample + 1, self.n_choices)] # [0,1,3,4]
            self.log_prob = torch.log(probs[sample])
            self.current_prob_over_ops = probs
            # set binary gate
            self.AP_path_wb.data[sample] = 1.0 # [0,0,1,0,0]
        # avoid over-regularization
        for _i in range(self.n_choices):
            for name, param in self.ops[_i].named_parameters():
                param.grad = None

    def set_arch_param_grad(self):
        binary_grads = self.AP_path_wb.grad.data
        # print(self.AP_path_wb.grad)
        # if self.active_op.is_zero_layer():
        #     self.AP_path_alpha.grad = None
        #     return
        # print("!!", self.AP_path_alpha)
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        # if MixedEdge.MODE == 'two':
        involved_idx = self.active_index + self.inactive_index
        probs_slice = F.softmax(torch.stack([
            self.AP_path_alpha[idx] for idx in involved_idx
        ]), dim=0).data
        for i in range(2):
            for j in range(2):
                origin_i = involved_idx[i]
                origin_j = involved_idx[j]
                self.AP_path_alpha.grad.data[origin_i] += \
                    binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i])
        # print(self.active_index)
        for _i, idx in enumerate(self.active_index):
            self.active_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
        # print(self.inactive_index)
        for _i, idx in enumerate(self.inactive_index):
            self.inactive_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
        # else:
        #     probs = self.probs_over_ops.data
        #     for i in range(self.n_choices):
        #         for j in range(self.n_choices):
        #             self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        # print(involved_idx)
        # print(old_alphas)
        # print(new_alphas)
        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset


class FBNet_Stochastic_SuperNet(nn.Module):
    def __init__(self, lookup_table):
        super(FBNet_Stochastic_SuperNet, self).__init__()

        self.current_img_layer = ConvBNRelu(input_depth=3, output_depth=16, kernel=7, stride=1,
                                pad=3, no_bias=1, use_relu="relu", bn_type="bn")

        self.previous_img_layer = ConvBNRelu(input_depth=3, output_depth=16, kernel=7, stride=1,
                                pad=3, no_bias=1, use_relu="relu", bn_type="bn")

        self.previous_hm_layer = ConvBNRelu(input_depth=1, output_depth=16, kernel=7, stride=1,
                                pad=3, no_bias=1, use_relu="relu", bn_type="bn")

        # self.first identical to 'add_first' in the fbnet_building_blocks/fbnet_builder.py
        self.first = ConvBNRelu(input_depth=16, output_depth=16, kernel=3, stride=2,
                                pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn") # in_dep=7

        self.stages_to_search = nn.ModuleList([MixedOperation(
                                                   lookup_table.layers_parameters[layer_id],
                                                   lookup_table.lookup_table_operations,
                                                   lookup_table.lookup_table_latency[layer_id])
                                               for layer_id in range(lookup_table.cnt_layers)])

        self.neck = nn.ModuleList([MixedOperation(
            lookup_table.layers_parameters_neck[layer_id],
            lookup_table.lookup_neck_operations,
            lookup_table.lookup_neck_latency[layer_id])
            for layer_id in range(lookup_table.cnt_layers_neck)])

        self.head = nn.ModuleList([MixedOperation(
            lookup_table.layers_parameters_head[layer_id],
            lookup_table.lookup_head_operations,
            lookup_table.lookup_head_latency[layer_id])
            for layer_id in range(lookup_table.cnt_layers_head)])

        self.preprocess = ConvBNRelu(input_depth=1024,
                                     output_depth=512,
                                     kernel=1, stride=1,
                                     pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.upsample = Upsample(scale_factor=2, mode="nearest")

        self.convert = ConvBNRelu(input_depth=512,
                                  output_depth=256,
                                  kernel=1, stride=1,
                                  pad=0, no_bias=1, use_relu="relu", bn_type="bn")
        
        out_layers_list = []
        self.out_heads = []
        for out_head, out_chns in CONFIG_SUPERNET['output_heads'].items():
            out_layer = nn.Conv2d(256, out_chns,
                                  kernel_size=1, stride=1, padding=0, bias=True)
            out_layers_list.append(out_layer)
            self.out_heads.append(out_head)
        self.out_layers = nn.ModuleList(out_layers_list)
            

    def forward(self, x, latency_to_accumulate):
        img = x['image']
        pre_img = x['pre_img']
        pre_hm = x['pre_hm']
        ins = self.current_img_layer(img)
        # print("step2")
        ins = ins + self.previous_img_layer(pre_img)
        # print("step3")
        ins = ins + self.previous_hm_layer(pre_hm)
        # print("step4")

        y = self.first(ins)
        for idx, mixed_op in enumerate(self.stages_to_search):
            # print("step6")
            y, latency_to_accumulate = mixed_op(y, latency_to_accumulate)
            if idx == 6:
                y_8 = y  # downratio8
        
        y = self.preprocess(y)  # downratio16
        y_fpn0, latency_to_accumulate = self.neck[0](y_8, latency_to_accumulate)
        y_fpn1, latency_to_accumulate = self.neck[1](y_8, latency_to_accumulate)
        y_fpn2, latency_to_accumulate = self.neck[2](y, latency_to_accumulate)
        y_fpn3, latency_to_accumulate = self.neck[3](y, latency_to_accumulate)
        y_fpnhid0 = y_fpn0 + self.upsample(y_fpn2)
        y_fpnhid1 = y_fpn1 + y_fpn3
        y_fpn4, latency_to_accumulate = self.neck[4](y_fpnhid0, latency_to_accumulate)
        y_fpn5, latency_to_accumulate = self.neck[5](y_fpnhid1, latency_to_accumulate)
        y = y_fpn4 + y_fpn5
        y = self.convert(y)
    
        for mixed_head_op in self.head:
            y, latency_to_accumulate = mixed_head_op(y, latency_to_accumulate)
            
        # y = self.last_stages(y)
        out = []
        z = {}
        # for out_head, out_chns in CONFIG_SUPERNET['output_heads'].items():
        #     out_layer = nn.Conv2d(256, out_chns,
        #                           kernel_size=1, stride=1, padding=0, bias=True)
        #     z[head] = out_layer(y)
        for ixx, out_layer in enumerate(self.out_layers):
            z[self.out_heads[ixx]] = out_layer(y)
            
        out.append(z)

        return out, latency_to_accumulate

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        # for m in self.redundant_modules:
        #     try:
        #         m.binarize()
        #     except AttributeError:
        #         print(type(m), ' do not support binarize')
        for m_b in self.stages_to_search:
            m_b.binarize()
        for m_n in self.neck:
            m_n.binarize()
        for m_h in self.head:
            m_h.binarize()

    # def unused_modules_off(self):
    #     self._unused_modules = []
    #     self.unused_modules_off_part(self.stages_to_search)
    #     self.unused_modules_off_part(self.neck)
    #     self.unused_modules_off_part(self.head)
    #
    # def unused_modules_off_part(self, redundant_modules):
    #     for m in redundant_modules:
    #         unused = {}
    #         if MixedEdge.MODE in ['full', 'two', 'full_v2']:
    #             involved_index = m.active_index + m.inactive_index
    #         else:
    #             involved_index = m.active_index
    #         for i in range(m.n_choices):
    #             if i not in involved_index:
    #                 unused[i] = m.ops[i]
    #                 m.ops[i] = None
    #         self._unused_modules.append(unused)

    def unused_modules_off(self):
        self._unused_modules_backbone = self.unused_modules_off_part(self.stages_to_search)
        self._unused_modules_neck = self.unused_modules_off_part(self.neck)
        self._unused_modules_head = self.unused_modules_off_part(self.head)

    def unused_modules_off_part(self, redundant_modules):
        unused_modules = []
        for m in redundant_modules:
            unused = {}
            # if MixedEdge.MODE in ['full', 'two', 'full_v2']:
            involved_index = m.active_index + m.inactive_index
            # else:
                # involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.ops[i]
                    m.ops[i] = None
            # self._unused_modules.append(unused)
            unused_modules.append(unused)
        return unused_modules

    def set_arch_param_grad(self):
        self.set_arch_param_grad_part(self.stages_to_search)
        self.set_arch_param_grad_part(self.neck)
        self.set_arch_param_grad_part(self.head)

    def set_arch_param_grad_part(self, redundant_modules):
        for m in redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        self.rescale_updated_arch_param_part(self.stages_to_search)
        self.rescale_updated_arch_param_part(self.neck)
        self.rescale_updated_arch_param_part(self.head)

    def rescale_updated_arch_param_part(self, redundant_modules):
        for m in redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    def unused_modules_back(self):
        self.unused_modules_back_part(self.stages_to_search, self._unused_modules_backbone)
        self.unused_modules_back_part(self.neck, self._unused_modules_neck)
        self.unused_modules_back_part(self.head, self._unused_modules_head)
        self._unused_modules_backbone = None
        self._unused_modules_head = None
        self._unused_modules_neck = None

    def unused_modules_back_part(self, redundant_modules, unused_modules):
        if unused_modules is None:
            return
        for m, unused in zip(redundant_modules, unused_modules):
            for i in unused:
                m.ops[i] = unused[i]

    def set_chosen_op_active(self):
        self.set_chosen_op_active_part(self.stages_to_search)
        self.set_chosen_op_active_part(self.neck)
        self.set_chosen_op_active_part(self.head)

    def set_chosen_op_active_part(self, redundant_modules):
        for m in redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')


class SupernetLoss(nn.Module):
    def __init__(self, opt):
        super(SupernetLoss, self).__init__()
        self.alpha = CONFIG_SUPERNET['loss']['alpha']
        self.beta = CONFIG_SUPERNET['loss']['beta']
        # self.opt = optss()
        self.opt = opt
        self.track_criterion = GenericLoss(self.opt)


    def forward(self, outs, targets, latency, losses_ce, losses_lat, N):
        ce, track_loss = self.track_criterion(outs, targets)
        # ce, track_loss = GenericLoss(outs, targets, self.opt)
        lat = torch.log(latency ** self.beta)

        losses_ce.update(ce.item(), N)
        losses_lat.update(lat.item(), N)

        loss = self.alpha * ce * lat # ce or track_loss? solved
        return loss, track_loss  # .unsqueeze(0)    currently undefined: what loss to add/use  -  solved


# from supernet_functions.lookup_table_builder import LookUpTable
# tbb = LookUpTable()
# anet = FBNet_Stochastic_SuperNet(tbb)
# for m in anet.children():
#     print(m)
# ### builder row 251

