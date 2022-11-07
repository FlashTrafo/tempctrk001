from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np

from general_functions.trackloss.losses import FastFocalLoss, RegWeightedL1Loss

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _sigmoid_output(output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    return output

def GenericLoss(outputs, batch, opt):
    crit = FastFocalLoss()
    crit_reg = RegWeightedL1Loss()
    losses = {head: 0 for head in opt.heads}

    for s in range(opt.num_stacks):
      output = outputs[s]
      output = _sigmoid_output(output)

      if 'hm' in output:
        losses['hm'] += self.crit(
          output['hm'], batch['hm'], batch['ind'],
          batch['mask'], batch['cat']) / opt.num_stacks
        print("!!!!!!!!!!!!", losses['hm'])

      regression_heads = [
        'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
        'dep', 'dim', 'amodel_offset', 'velocity']

      for head in regression_heads:
        if head in output:
          losses[head] += crit_reg(
            output[head], batch[head + '_mask'],
            batch['ind'], batch[head]) / opt.num_stacks

      if 'hm_hp' in output:
        losses['hm_hp'] += crit(
          output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
          batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
        if 'hp_offset' in output:
          losses['hp_offset'] += crit_reg(
            output['hp_offset'], batch['hp_offset_mask'],
            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

    losses['tot'] = 0
    for head in opt.heads:
      losses['tot'] += opt.weights[head] * losses[head]

    return losses['tot'], losses
    