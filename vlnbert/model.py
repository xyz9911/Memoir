import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from vlnbert.vlnbert_init import get_vlnbert_models

class VLNBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)

    def forward(self, mode, batch):
        batch = collections.defaultdict(lambda: None, batch)
        
        if mode == 'language':            
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds

        elif mode == 'panorama':
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            pano_embeds, pano_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            outs = self.vln_bert(mode, batch)
            return outs

        elif mode == 'init':
            outs = self.vln_bert(mode, batch)
            return outs

        elif mode == 'belief':
            outs = self.vln_bert(mode, batch)
            return outs

        elif mode == 'transform':
            outs = self.vln_bert(mode, batch)
            return outs

        elif mode == 'transit':
            outs = self.vln_bert(mode, batch)
            return outs

        elif mode == 'imagine':
            outs = self.vln_bert(mode, batch)
            return outs

        elif mode == 'reconstruct':
            outs = self.vln_bert(mode, batch)
            return outs

        elif mode == 'memory':
            outs = self.vln_bert(mode, batch)
            return outs
        
        elif mode == 'posterior':
            outs = self.vln_bert(mode, batch)
            return outs

        else:
            raise NotImplementedError('wrong mode: %s'%mode)
