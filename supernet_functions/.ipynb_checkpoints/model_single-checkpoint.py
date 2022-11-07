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


class MixedOperation(nn.Module):
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
        # self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))
        # self.log_prob = None
        # self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))
        # self._unused_modules_backbone = []
        # self._unused_modules_neck = []
        # self._unused_modules_head = []
        # self.active_index = [0]
        # self.inactive_index = None
        # self.current_prob_over_ops = None

    def forward(self, x, latency_to_accumulate):
        output = self.ops[0](x)
        latency_to_accumulate = latency_to_accumulate + self.latency[0]

        return output, latency_to_accumulate


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

