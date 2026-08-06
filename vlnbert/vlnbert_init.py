# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

from transformers import (BertConfig, BertTokenizer)
from vlnbert.vilmodel import GlocalTextPathNavCMT
from transformers import PretrainedConfig
import copy
import torch
import os


def get_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def get_vlnbert_models(args, config=None):
    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        for k, v in ckpt_weights.items():
            if 'global_encoder' in k:
                new_k = k.replace('global_encoder', 'exp_encoder')
                new_ckpt_weights[new_k] = copy.deepcopy(v)
            elif 'global_sap_head' in k:
                new_k = k.replace('global_sap_head', 'exp_sap_head')
                new_ckpt_weights[new_k] = copy.deepcopy(v)
            new_ckpt_weights[k] = v

    if args.tokenizer == 'xlm':
        cfg_name = os.path.join('lxmert-base-uncased')
    else:
        cfg_name = os.path.join('bert-base-uncased')
    vis_config = PretrainedConfig.from_pretrained(cfg_name)
    
    vis_config.world_model_type = 'tssm'

    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.graph_sprels = args.graph_sprels
    vis_config.glocal_fuse = args.fusion == 'dynamic'

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_pano_embedding = args.fix_pano_embedding
    vis_config.fix_local_branch = args.fix_local_branch

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False

    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None,
        config=vis_config,
        state_dict=new_ckpt_weights)

    return visual_model
