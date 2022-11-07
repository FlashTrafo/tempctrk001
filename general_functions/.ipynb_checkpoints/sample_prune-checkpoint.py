from supernet_functions.lookup_table_builder import *
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Upsample, Identity, Zero
import copy
import torch
import torch.nn as nn
from supernet_main_file import sample_architecture_from_the_supernet
import math
from general_functions.trackloss.trackgenericloss import GenericLoss

sampled, sampled_idx, state_dict = sample_architecture_from_the_supernet('testnet001')
residuals = [5, 8, 10, 12, 15, 17, 19, 20, 21, 22, 23]
pure_skip = []
cnt_backbone = 11
cnt_fpn = 6
cnt_head = 5
no_preprocess = 11
no_convert = 18
f0 = 12
b0 = 6
h0 = 19
blks = []

for i in residuals:
    if sampled[i] == 'skip':
        pure_skip.append(i)  # residual + skip
        
for i, j in enumerate(sampled):
    if j != 'skip' and j != 'none' and j != 'no':
        blks.append(i)

bn_weights = []
bn_weights_avg = []

# read bn weights
for i in range(len(sampled)):
    bn_paras = {}
    if i < cnt_backbone:
        if sampled[i] != 'skip':
            bn_paras['pw_w'] = state_dict['stages_to_search.{}.ops.{}.pw.bn.weight'.format(i, sampled_idx[i])]  # scaling factor
            bn_paras['dw_w'] = state_dict['stages_to_search.{}.ops.{}.dw.bn.weight'.format(i, sampled_idx[i])]
            bn_paras['pwl_w'] = state_dict['stages_to_search.{}.ops.{}.pwl.bn.weight'.format(i, sampled_idx[i])]
        else:
            if i not in pure_skip:
                # bn_paras['pwl_w'] = state_dict['stages_to_search.{}.ops.10.conv.bn.weight'.format(i)] # "regarded" as pwl_w so that same name
                 bn_paras['pwl_w'] = state_dict['stages_to_search.{}.ops.{}.conv.bn.weight'.format(i, sampled_idx[i])]

    elif i == no_preprocess:
        bn_paras['pwl_w'] = state_dict['preprocess.bn.weight']

    elif i > cnt_backbone and i < cnt_backbone + cnt_fpn + 1:
        if sampled[i] != 'skip' and sampled[i] != 'none':
            bn_paras['pw_w'] = state_dict['neck.{}.ops.{}.pw.bn.weight'.format(i - f0, sampled_idx[i])]
            bn_paras['dw_w'] = state_dict['neck.{}.ops.{}.dw.bn.weight'.format(i - f0, sampled_idx[i])]
            bn_paras['pwl_w'] = state_dict['neck.{}.ops.{}.pwl.bn.weight'.format(i - f0, sampled_idx[i])]
        elif sampled[i] == 'skip':
            if i not in pure_skip:
                bn_paras['pwl_w'] = state_dict['neck.{}.ops.7.conv.bn.weight'.format(i - f0)]

    elif i == no_convert:
        bn_paras['pwl_w'] = state_dict['convert.bn.weight']

    else:  # head
        if sampled[i] != 'skip':
            bn_paras['pw_w'] = state_dict['head.{}.ops.{}.pw.bn.weight'.format(i - h0, sampled_idx[i])]
            bn_paras['dw_w'] = state_dict['head.{}.ops.{}.dw.bn.weight'.format(i - h0, sampled_idx[i])]
            bn_paras['pwl_w'] = state_dict['head.{}.ops.{}.pwl.bn.weight'.format(i - h0, sampled_idx[i])]

    bn_weights.append(bn_paras)

bn_weights_avg = copy.deepcopy(bn_weights)

# prune
total = 0
for layer in bn_weights:
    for m in layer.values():
        total += m.shape[0] # num of channels

# synchrone
group_tot = 0
pwdwtot = 0
group_ids = [0] * len(sampled)

for i in range(cnt_backbone - 1): # backbone
    if sampled[i] == 'skip':
        if i in pure_skip:
            continue
    k = i + 1
    while 1:
        if sampled[k] == 'skip':
            if k in pure_skip and k < cnt_backbone:
                k += 1
                continue
        if k in residuals:
            if group_ids[i] == 0:
                group_tot += 1
                group_ids[i] = group_tot
                group_ids[k] = group_tot
            else:
                group_ids[k] = group_ids[i]
        break

def synfpn(i, j, group_tot):
    if sampled[i] != 'none' and i not in pure_skip and sampled[j] != 'none' and j not in pure_skip:
        if group_ids[i] == 0:
            group_tot += 1
            group_ids[i] = group_tot
            group_ids[j] = group_tot
        else:
            group_ids[j] = group_ids[i]
    return group_tot

# f0 = 12
# b0 = 6
group_tot = synfpn(b0, f0, group_tot)
group_tot = synfpn(f0, f0 + 2, group_tot)
if sampled[f0] == 'skip':
    group_tot = synfpn(b0, f0 + 2, group_tot)
group_tot = synfpn(f0 - 1, f0 + 3, group_tot)
group_tot = synfpn(f0 + 3, f0 + 1, group_tot)
group_tot = synfpn(f0 + 3, f0 + 5, group_tot)
group_tot = synfpn(f0 + 1, f0 + 5, group_tot)
group_tot = synfpn(f0 + 5, f0 + 4, group_tot)
if sampled[f0 + 3] == 'skip':
    group_tot = synfpn(f0 - 1, f0 + 1, group_tot)
    group_tot = synfpn(f0 - 1, f0 + 5, group_tot)
    group_tot = synfpn(f0 + 5, f0 + 4, group_tot)
if sampled[f0 + 5] == 'skip':
    group_tot = synfpn(f0 - 1, f0 + 4, group_tot)
    group_tot = synfpn(f0 + 1, f0 + 4, group_tot)

group_tot += 1
for i in range(no_convert, len(sampled)):
    if i not in pure_skip:
        group_ids[i] = group_tot

aver = []
# group finished, syn:
for i in range(group_tot):
    g = []
    for j, k in enumerate(group_ids):
        if k == i + 1:
            g.append(j)
    aver.append(g)
print(aver) 

# averaging
for g in aver:
    if g == []:
        continue
    gsum = torch.zeros(bn_weights[g[0]]['pwl_w'].shape).cuda('cuda:2')
    for i in g:
        # gsum += bn_paras[i]['pwl_w']
        gsum += bn_weights[i]['pwl_w']
    gsum /= len(g)
    for i in g:
        bn_weights_avg[i]['pwl_w'] = gsum
        
for w in blks:
    pd = torch.zeros(bn_weights[w]['pw_w'].shape).cuda('cuda:2')
    pd += bn_weights[w]['pw_w']
    pd += bn_weights[w]['dw_w']
    pd /= 2
    bn_weights_avg[w]['pw_w'] = pd
    bn_weights_avg[w]['dw_w'] = pd

# finish syn
bn = torch.zeros(total).cuda()
index = 0
for layer in bn_weights_avg:
    for m in layer.values():
        size = m.shape[0]
        bn[index:(index+size)] = m.abs().clone()
        index += size

# prune
percent = 0.5

y, i = torch.sort(bn)
thre_index = int(total * percent)  # threshold index
thre = y[thre_index].cuda('cuda:2')

pruned = 0
result_channels = []  # dicts: {convlayer: preserved channel idx}
for layer in bn_weights_avg:
    res = {}
    for k, m in layer.items():
        mask = m.abs().gt(thre) #.cuda()
        remain_this = torch.sum(mask)
        pruned = pruned + mask.shape[0] - remain_this
        print(remain_this)
        if remain_this > 0:
            # pruned = pruned + mask.shape[0] - torch.sum(mask)
            preserved = torch.nonzero(mask).squeeze()
            res[k] = preserved
        else:
            pres_val, pres_idx = torch.max(m, 0)
            res[k] = pres_idx
            pruned -= 1
    result_channels.append(res)

pruned_ratio = pruned/total #????????

inputs = [0] * len(sampled)
inputs[0] = 16
inpdir = [-1] * len(sampled)
for i in range(1, cnt_backbone + 1):
    if i - 1 in pure_skip:
        inputs[i] = inputs[i - 1]  # no skip-skip exists
        inpdir[i] = i - 2
    else:
        inputs[i] = len(result_channels[i - 1]['pwl_w'])
        inpdir[i] = i - 1
inputs[f0] = len(result_channels[b0]['pwl_w'])
inpdir[f0] = b0
inputs[f0 + 1] = len(result_channels[b0]['pwl_w'])
inpdir[f0 + 1] = b0
inputs[f0 + 2] = len(result_channels[f0 - 1]['pwl_w'])
inpdir[f0 + 2] = f0 - 1
inputs[f0 + 3] = len(result_channels[f0 - 1]['pwl_w'])
inpdir[f0 + 3] = f0 - 1

if sampled[f0] == 'none' or f0 in pure_skip:
    if sampled[f0 + 2] == 'none':
        inputs[f0 + 4] = inputs[f0]
        inpdir[f0 + 4] = b0
    else:
        inputs[f0 + 4] = len(result_channels[f0 + 2]['pwl_w'])
        inpdir[f0 + 4] = f0 + 2
else:
    inputs[f0 + 4] = len(result_channels[f0]['pwl_w'])
    inpdir[f0 + 4] = f0

case0 = 0
if sampled[f0 + 3] == 'none' or f0 + 3 in pure_skip:
    if sampled[f0 + 1] == 'none':
        inputs[f0 + 5] = inputs[f0 + 3]
        inpdir[f0 + 5] = f0 - 1
    else:
        inputs[f0 + 5] = len(result_channels[f0 + 1]['pwl_w'])
        inpdir[f0 + 5] = f0 + 1
        case0 = 1
else:
    inputs[f0 + 5] = len(result_channels[f0 + 3]['pwl_w'])
    inpdir[f0 + 5] = f0 + 3
    case0 = 2
    
if sampled[f0 + 5] == 'none' or f0 + 5 in pure_skip:
    if sampled[f0 + 4] == 'none':
        inputs[no_convert] = inputs[f0 + 5]
        if case0 == 0: ## redundant code - not needed because already syned
            inpdir[no_convert] = f0 - 1
        elif case0 == 1:
            inpdir[no_convert] = f0 + 1
        else:
            inpdir[no_convert] = f0 + 3
    else:
        inputs[no_convert] = len(result_channels[f0 + 4]['pwl_w'])
        inpdir[no_convert] = f0 + 4
else:
    inputs[no_convert] = len(result_channels[f0 + 5]['pwl_w'])
    inpdir[no_convert] = f0 + 5

for i in range(no_convert + 1, len(sampled)):
    inputs[i] = len(result_channels[no_convert]['pwl_w'])
    inpdir[i] = no_convert
    
strides = []
strides = strides + SEARCH_SPACE['strides']
strides.append(1)
strides = strides + NECK_SEARCH_SPACE['strides']
strides.append(1)
strides = strides + HEAD_SEARCH_SPACE['strides']

pruned_layers_parameters = []
for i in range(len(sampled)):
    if sampled[i] != 'skip' and sampled[i] != 'none' and sampled[i] != 'no':
        pruned_layers_parameters.append((inputs[i],
                                         len(result_channels[i]['pwl_w']),
                                         len(result_channels[i]['pw_w']),
                                         strides[i]))
    elif sampled[i] == 'skip' or sampled[i] == 'no':
        if i not in pure_skip:
            pruned_layers_parameters.append((inputs[i],
                                             len(result_channels[i]['pwl_w']),
                                             strides[i]))    # num of chns: fpn: min of two plussers
        else:
            pruned_layers_parameters.append((inputs[i],
                                             inputs[i],
                                             strides[i]))
    else: # none
        pruned_layers_parameters.append((strides[i],))

print(pruned_layers_parameters)


class Pruned_IRFBlock(nn.Module):
    def __init__(
            self,
            input_depth,
            output_depth,
            stride,
            mid_depth,
            bn_type="bn",
            kernel=3,
            nl="relu",
            dil=1,
            width_divisor=1,
            shuffle_type=None,
            pw_group=1,
            se=False,
            dw_skip_bn=False,
            dw_skip_relu=False,
    ):
        super(Pruned_IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        # mid_depth = int(input_depth * expansion)
        # mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=nl,
            bn_type=bn_type,
            group=pw_group,
        )

        # dw
        self.dw = ConvBNRelu(
            mid_depth,
            mid_depth,
            kernel=kernel,
            stride=stride,
            pad=(kernel // 2) * dil,
            dil=dil,
            group=mid_depth,
            no_bias=1,
            use_relu=nl if not dw_skip_relu else None,
            bn_type=bn_type if not dw_skip_bn else None,
        )

        # pw-linear
        self.pwl = ConvBNRelu(
            mid_depth,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=False,
            bn_type=bn_type,
            group=pw_group,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.se4 = SEModule(output_depth) if se else nn.Sequential()

        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)

        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y


PRIMITIVES = {
    "none": lambda stride, **kwargs: Zero(
        stride
    ),
    "skip": lambda C_in, C_out, stride, **kwargs: Identity(
        C_in, C_out, stride
    ),
    "ir_k3_re": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=3, nl="relu", **kwargs
    ),
    "ir_k3_hs": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=3, nl="hswish", **kwargs
    ),
    "ir_k3_r2_re": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=3, nl="relu", dil=2, **kwargs
    ),
    "ir_k3_r2_hs": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=3, nl="hswish", dil=2, **kwargs
    ),
    "ir_k3_r3_re": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=3, nl="relu", dil=3, **kwargs
    ),
    "ir_k5_re": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=5, nl="relu", **kwargs
    ),
    "ir_k5_hs": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=5, nl="hswish", **kwargs
    ),
    "ir_k5_r2_re": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=5, nl="relu", dil=2, **kwargs
    ),
    "ir_k5_r2_hs": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=5, nl="hswish", dil=2, **kwargs
    ),
    "ir_k5_r3_re": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=5, nl="relu", dil=3, **kwargs
    ),
    "ir_k7_re": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=7, nl="relu", **kwargs
    ),
    "ir_k7_hs": lambda C_in, C_out, mid_depth, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth, kernel=7, nl="hswish", **kwargs
    ),
}


class Sampled_pruned_net(nn.Module):
    def __init__(self, paras):
        super(Sampled_pruned_net, self).__init__()

        # self.first identical to 'add_first' in the fbnet_building_blocks/fbnet_builder.py
        self.first = ConvBNRelu(input_depth=16, output_depth=16, kernel=3, stride=2,
                                pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn")
        self.paras = paras
        # didn't add names to sequential
        self.backbone = nn.ModuleList()
        for i in range(cnt_backbone):
            self.backbone.append(PRIMITIVES[sampled[i]](*self.paras[i]))

        self.neck = nn.ModuleList()
        for i in range(f0, f0 + cnt_fpn):
            self.neck.append(PRIMITIVES[sampled[i]](*self.paras[i]))

        self.head = nn.ModuleList()
        for i in range(no_convert + 1, len(sampled)):
            self.head.append(PRIMITIVES[sampled[i]](*self.paras[i]))

        # self.preprocess = ConvBNRelu(*self.paras[no_preprocess])
        
        in11, out11, stride11 = self.paras[no_preprocess]
        self.preprocess = ConvBNRelu(input_depth=in11,
                             output_depth=out11,
                             kernel=1, stride=stride11,
                             pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.upsample = Upsample(scale_factor=2, mode="nearest")

        # self.convert = ConvBNRelu(*self.paras[no_convert])
        in18, out18, stride18 = self.paras[no_convert]
        self.convert = ConvBNRelu(input_depth=in18,
                                  output_depth=out18,
                                  kernel=1, stride=stride18,
                                  pad=0, no_bias=1, use_relu="relu", bn_type="bn")
        
        self.current_img_layer = ConvBNRelu(input_depth=3, output_depth=16, kernel=7, stride=1,
                                pad=3, no_bias=1, use_relu="relu", bn_type="bn")

        self.previous_img_layer = ConvBNRelu(input_depth=3, output_depth=16, kernel=7, stride=1,
                                pad=3, no_bias=1, use_relu="relu", bn_type="bn")

        self.previous_hm_layer = ConvBNRelu(input_depth=1, output_depth=16, kernel=7, stride=1,
                                pad=3, no_bias=1, use_relu="relu", bn_type="bn")
        
        out_layers_list = []
        self.out_heads = []
        for out_head, out_chns in CONFIG_SUPERNET['output_heads'].items():
            # out_layer = nn.Conv2d(256, out_chns,
            out_layer = nn.Conv2d(len(result_channels[no_convert]['pwl_w']), out_chns,
                                  kernel_size=1, stride=1, padding=0, bias=True)
            out_layers_list.append(out_layer)
            self.out_heads.append(out_head)
        self.out_layers = nn.ModuleList(out_layers_list)

    def forward(self, x):
        img = x['image']
        pre_img = x['pre_img']
        pre_hm = x['pre_hm']
        ins = self.current_img_layer(img)
        ins = ins + self.previous_img_layer(pre_img)
        ins = ins + self.previous_hm_layer(pre_hm)
        y = self.first(ins)

        for idx, layer in enumerate(self.backbone):
            y = layer(y)
            if idx == 6:
                y_8 = y

        y = self.preprocess(y)  # downratio16

        y_fpn0 = self.neck[0](y_8)
        y_fpn1 = self.neck[1](y_8)
        y_fpn2 = self.neck[2](y)
        y_fpn3 = self.neck[3](y)
        y_fpnhid0 = y_fpn0 + self.upsample(y_fpn2)
        y_fpnhid1 = y_fpn1 + y_fpn3
        y_fpn4 = self.neck[4](y_fpnhid0)
        y_fpn5 = self.neck[5](y_fpnhid1)
        y = y_fpn4 + y_fpn5

        y = self.convert(y)

        for head_layer in self.head:
            y = head_layer(y)
        # y = self.last_stages(y)
        out = []
        z = {}
        for ixx, out_layer in enumerate(self.out_layers):
            z[self.out_heads[ixx]] = out_layer(y)
        out.append(z)
        
        return out

# for i in result_channels:
#     for k, j in i.items():
#         print(k, j.shape)

def create_sampled_net():
    
    print(pruned_ratio)
    
    model = Sampled_pruned_net(pruned_layers_parameters)
    new_state_dict = model.state_dict()

    # k_else_1 = ['.pw.bn.weight', '.pw.bn.bias', '.pw.bn.running_mean', '.pw.bn.running_var',
    #             '.dw.bn.weight', '.dw.bn.bias', '.dw.bn.running_mean', '.dw.bn.running_var',
    #             '.pwl.bn.weight', '.pwl.bn.bias', '.pwl.bn.running_mean', '.pwl.bn.running_var',]
    # k_else_4 = ['.pw.conv.weight', '.pw.conv.quan_w_fn.s',
    #             '.dw.conv.weight', '.dw.conv.quan_w_fn.s'
    #             '.pwl.conv.weight', '.pwl.conv.quan_w_fn.s']
    # # k_else_0 = [] activation???
    # k_skip_1 = ['.conv.bn.weight', '.conv.bn.bias', '.conv.bn.running_mean', '.conv.bn.running_var']
    # k_skip_4 = ['.conv.conv.weight', '.conv.conv.quan_w_fn.s']
    
    k_else_1 = ['.pw.bn.weight', '.pw.bn.bias', '.pw.bn.running_mean', '.pw.bn.running_var',
                '.dw.bn.weight', '.dw.bn.bias', '.dw.bn.running_mean', '.dw.bn.running_var',
                '.pwl.bn.weight', '.pwl.bn.bias', '.pwl.bn.running_mean', '.pwl.bn.running_var', ]
    # k_else_4 = ['.pw.conv.weight',
    #             '.dw.conv.weight',
    #             '.pwl.conv.weight']
    k_41 = '.pw.conv.weight'
    k_42 = '.dw.conv.weight'
    k_43 = '.pwl.conv.weight'
    # k_else_0 = [] activation???
    k_skip_1 = ['.conv.bn.weight', '.conv.bn.bias', '.conv.bn.running_mean', '.conv.bn.running_var']
    k_skip_4 = ['.conv.conv.weight']
    
    kdic = ['pw.','dw.','pwl.']
    wdic = ['pw_w','dw_w','pwl_w']

    for i in range(cnt_backbone):
        k0 = 'backbone.{}'.format(i)
        k1 = 'stages_to_search.{}.ops.{}'.format(i, sampled_idx[i])
        if sampled[i] != 'skip':
            k = k0 + k_41
            p = k1 + k_41
            if i > 0:
                temp = state_dict[p][result_channels[i]['pw_w'], :, :, :]
                new_state_dict[k] = temp[:, result_channels[inpdir[i]]['pwl_w'], :, :]
            else:
                new_state_dict[k] = state_dict[p][result_channels[i]['pw_w'], :, :, :]
            k = k0 + k_42
            p = k1 + k_42
            new_state_dict[k] = state_dict[p][result_channels[i]['dw_w'], :, :, :]
            k = k0 + k_43
            p = k1 + k_43
            temp = state_dict[p][result_channels[i]['pwl_w'], :, :, :]
            new_state_dict[k] = temp[:, result_channels[i]['dw_w'], :, :]
            for key in k_else_1:
                k = k0 + key
                p = k1 + key
                for a0, b0 in enumerate(kdic):
                    if b0 in key:
                        new_state_dict[k] = state_dict[p][result_channels[i][wdic[a0]]]
        else:
            if i not in pure_skip:
                for key in k_skip_4:
                    k = k0 + key
                    p = k1 + key
                    temp = state_dict[p][result_channels[i]['pwl_w'], :, :, :]
                    new_state_dict[k] = temp[:, result_channels[inpdir[i]]['pwl_w'], :, :]
                for key in k_skip_1:
                    k = k0 + key
                    p = k1 + key
                    new_state_dict[k] = state_dict[p][result_channels[i]['pwl_w']]

    for i in range(cnt_fpn):
        k0 = 'neck.{}'.format(i)
        k1 = 'neck.{}.ops.{}'.format(i, sampled_idx[i + f0])
        if sampled[i + f0] != 'skip' and sampled[i + f0] != 'none':
            k = k0 + k_41
            p = k1 + k_41
            temp = state_dict[p][result_channels[i + f0]['pw_w'], :, :, :]
            new_state_dict[k] = temp[:, result_channels[inpdir[i + f0]]['pwl_w'], :, :]
            k = k0 + k_42
            p = k1 + k_42
            new_state_dict[k] = state_dict[p][result_channels[i + f0]['dw_w'], :, :, :]
            k = k0 + k_43
            p = k1 + k_43
            temp = state_dict[p][result_channels[i + f0]['pwl_w'], :, :, :]
            new_state_dict[k] = temp[:, result_channels[i + f0]['dw_w'], :, :]
            for key in k_else_1:
                k = k0 + key
                p = k1 + key
                # new_state_dict[k] = state_dict[p][result_channels[i + f0]['pwl_w']]
                for a0, b0 in enumerate(kdic):
                    if b0 in key:
                        new_state_dict[k] = state_dict[p][result_channels[i + f0][wdic[a0]]]
        elif sampled[i + f0] == 'skip':
            if i + f0 not in pure_skip:
                for key in k_skip_4:
                    k = k0 + key
                    p = k1 + key
                    temp = state_dict[p][result_channels[i + f0]['pwl_w'], :, :, :]
                    new_state_dict[k] = temp[:, result_channels[inpdir[i + f0]]['pwl_w'], :, :]
                for key in k_skip_1:
                    k = k0 + key
                    p = k1 + key
                    new_state_dict[k] = state_dict[p][result_channels[i + f0]['pwl_w']]

    for i in range(cnt_head):
        k0 = 'head.{}'.format(i)
        k1 = 'head.{}.ops.{}'.format(i, sampled_idx[i + no_convert + 1])
        if sampled[i + no_convert + 1] != 'skip':
            k = k0 + k_41
            p = k1 + k_41
            temp = state_dict[p][result_channels[i + no_convert + 1]['pw_w'], :, :, :]
            new_state_dict[k] = temp[:, result_channels[inpdir[i + no_convert + 1]]['pwl_w'], :, :]
            k = k0 + k_42
            p = k1 + k_42
            new_state_dict[k] = state_dict[p][result_channels[i + no_convert + 1]['dw_w'], :, :, :]
            k = k0 + k_43
            p = k1 + k_43
            temp = state_dict[p][result_channels[i + no_convert + 1]['pwl_w'], :, :, :]
            new_state_dict[k] = temp[:, result_channels[i + no_convert + 1]['dw_w'], :, :]
            for key in k_else_1:
                k = k0 + key
                p = k1 + key
                # new_state_dict[k] = state_dict[p][result_channels[i + no_convert + 1]['pwl_w']]
                for a0, b0 in enumerate(kdic):
                    if b0 in key:
                        new_state_dict[k] = state_dict[p][result_channels[i + no_convert + 1][wdic[a0]]]
        # else:
        #     for key in k_skip_4:
        #         k = k0 + key
        #         p = k1 + key
        #         new_state_dict[k] = state_dict[p][result_channels[i + no_convert + 1]['pwl_w'], :, :, :]
        #     for key in k_skip_1:
        #         k = k0 + key
        #         p = k1 + key
        #         new_state_dict[k] = state_dict[p][result_channels[i + no_convert + 1]['pwl_w']]
    # raise NotImplementedError
    for k, v in state_dict.items():
        # if 'preprocess.conv' in k:
        #     new_state_dict[k] = state_dict[k][result_channels[no_preprocess], :, :, :]
        if 'preprocess.bn' in k:
            if 'num' not in k:
                new_state_dict[k] = state_dict[k][result_channels[no_preprocess]['pwl_w']]
        # if 'convert.conv' in k:
        #     new_state_dict[k] = state_dict[k][result_channels[no_convert], :, :, :]
        if 'convert.bn' in k:
            if 'num' not in k:
                new_state_dict[k] = state_dict[k][result_channels[no_convert]['pwl_w']]
    
        if 'first' in k:
            new_state_dict[k] = state_dict[k]
        if 'layer' in k:
            if 'out' in k:
                if 'bias' in k:
                    new_state_dict[k] = state_dict[k]
                else:
                    new_state_dict[k] = state_dict[k][:, result_channels[no_convert]['pwl_w'], :, :]
            else:
                new_state_dict[k] = state_dict[k]
            
    tempp = state_dict['preprocess.conv.weight'][result_channels[no_preprocess]['pwl_w'], :, :, :]
    new_state_dict['preprocess.conv.weight'] = tempp[:, result_channels[inpdir[no_preprocess]]['pwl_w'], :, :]
    tempc = state_dict['convert.conv.weight'][result_channels[no_convert]['pwl_w'], :, :, :]
    new_state_dict['convert.conv.weight'] = tempc[:, result_channels[inpdir[no_convert]]['pwl_w'], :, :]
    # convd = ['.bn.weight', '.bn.bias', '.bn.running_mean', '.bn.running_var']
    # pfc = ['preprocess', 'convert']
    # for i in pfc:
    #     for j in convd:
    #         k = i + j
    #         q = 'no_' + i
    #         new_state_dict[k] = state_dict[result_channels[q]['pwl']]
                               
    model.load_state_dict(new_state_dict)

    return model


class SampledLoss(nn.Module):
    def __init__(self, opt):
        super(SampledLoss, self).__init__()
        self.opt = opt
        self.track_criterion = GenericLoss(self.opt)

    def forward(self, outs, targets):
        loss, track_loss = self.track_criterion(outs, targets)
        return loss, track_loss