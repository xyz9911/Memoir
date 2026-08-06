import copy
import random
from collections import defaultdict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .ops import pad_tensors_wgrad, gen_seq_masks, zdistr, truncate_text_into_chunks
from .vilmodel import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT
from .graph_utils import GraphMap, get_angle_fts


class RegionClassification(nn.Module):
    " for MRC(-kl)"

    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


def info_nce_loss(z, x, labels=None, temperature=0.05, reduction='mean', normalize=True):
    if normalize:
        z = F.normalize(z, dim=1)
        x = F.normalize(x, dim=1)
    logits = torch.matmul(z, x.t())
    if labels is None:
        labels = torch.arange(len(z), device=z.device)
    info_nce = F.cross_entropy(logits / temperature, labels, reduction=reduction)

    # penalty_mask = sources.unsqueeze(1) == extended_sources.unsqueeze(0)
    # diag_mask = labels.unsqueeze(1) != torch.arange(len(extended_sources), device=extended_sources.device).unsqueeze(0)
    # penalty_mask = torch.logical_and(penalty_mask, diag_mask)
    # masked_logits = torch.where(penalty_mask, torch.full_like(logits, -float('inf')), logits)
    # info_nce += F.cross_entropy(masked_logits / temperature, labels, reduction=reduction) * penalty_weight
    return info_nce

class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)

        self.drop_env = nn.Dropout(p=config.env_drop_prob)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
            if self.config.obj_prob_size > 0 and self.config.obj_prob_size != self.config.image_prob_size:
                self.obj_classifier = RegionClassification(self.config.hidden_size, self.config.obj_prob_size)
            else:
                self.obj_classifier = None
        if 'sap' in config.pretrain_tasks:
            self.global_sap_head = ClsPrediction(self.config.hidden_size)
            self.local_sap_head = ClsPrediction(self.config.hidden_size)
            if config.glocal_fuse:
                self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size * 2)
            else:
                self.sap_fuse_linear = None
        # todo: VAE
        if 'sr' in config.pretrain_tasks:
            pass

        self.init_weights()
        self.tie_weights()

        if config.fix_pano_embedding:
            for k, v in self.bert.img_embeddings.named_parameters():
                v.requires_grad = False

        if config.fix_lang_encoder:
            for k, v in self.bert.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.bert.lang_encoder.named_parameters():
                v.requires_grad = False

        if config.fix_local_branch:
            for k, v in self.bert.local_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.bert.local_sap_head.named_parameters():
                v.requires_grad = False

        if config.fix_global_branch:
            for k, v in self.bert.global_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.bert.global_sap_head.named_parameters():
                v.requires_grad = False
            for k, v in self.bert.sap_fuse_linear.named_parameters():
                v.requires_grad = False

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True, nav_db=None):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_node_types'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['txt_labels'], compute_loss
            )
        elif task.startswith('mrc'):
            return self.forward_mrc(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_node_types'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['vp_view_mrc_masks'], batch['vp_view_probs'],
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], compute_loss
            )
        elif task.startswith('sap'):
            return self.forward_sap(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_node_types'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'],
                batch['global_act_labels'], batch['local_act_labels'], compute_loss
            )
        elif task.startswith('sr'):
            return self.forward_sr(
                batch['txt_ids'], batch['txt_lens'], batch['scans'],
                batch['sent_embeds'], batch['full_sent_embeds'], batch['headings'], batch['paths'],
                batch['gt_paths'], nav_db, compute_loss
            )
        elif task.startswith('transformer'):
            return self.forward_transformer(
                batch['txt_ids'], batch['txt_lens'], batch['scans'], batch['headings'], batch['gt_paths'], nav_db,
                compute_loss
            )
        elif task == 'rssm':
            return self.forward_rssm(
                batch['txt_ids'], batch['txt_lens'], batch['scans'], batch['headings'], batch['gt_paths'], nav_db,
                compute_loss, batch['policy_weight'], batch['kl_weight'], batch['recon_weight'], batch['kl_balancing']
            )
        else:
            raise ValueError('invalid task')

    # Todo: remove state feature in language modeling
    def forward_mlm(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_node_types, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            txt_labels, compute_loss
    ):
        txt_embeds = self.bert.forward_mlm(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_node_types, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction='none'
            )
            return mask_loss.mean()
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    # Todo: remove state feature in region classification
    def forward_mrc(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_node_types, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            vp_view_mrc_masks, vp_view_probs, vp_obj_mrc_masks, vp_obj_probs, compute_loss=True
    ):

        _, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_node_types, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens)]
        vp_view_embeds = pad_tensors_wgrad(
            [x[1:view_len + 1] for x, view_len in zip(vp_embeds, vp_view_lens)]
        )  # [stop] at 0
        # vp_view_mrc_masks = vp_view_mrc_masks[:, :vp_view_embeds.size(1)]

        # only compute masked regions for better efficient=cy
        view_masked_output = self._compute_masked_hidden(vp_view_embeds, vp_view_mrc_masks)
        view_prediction_soft_labels = self.image_classifier(view_masked_output)
        view_mrc_targets = self._compute_masked_hidden(vp_view_probs, vp_view_mrc_masks)

        if traj_obj_img_fts is not None:
            vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens)]
            vp_obj_embeds = pad_tensors_wgrad(
                [x[view_len + 1:view_len + obj_len + 1] for x, view_len, obj_len in
                 zip(vp_embeds, vp_view_lens, vp_obj_lens)]
            )
            # vp_obj_mrc_masks = vp_obj_mrc_masks[:, :vp_obj_embeds.size(1)]
            obj_masked_output = self._compute_masked_hidden(vp_obj_embeds, vp_obj_mrc_masks)
            if self.obj_classifier is None:
                obj_prediction_soft_labels = self.image_classifier(obj_masked_output)
            else:
                obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
            obj_mrc_targets = self._compute_masked_hidden(vp_obj_probs, vp_obj_mrc_masks)
        else:
            obj_prediction_soft_labels, obj_mrc_targets = None, None

        if compute_loss:
            view_prediction_soft_labels = F.log_softmax(view_prediction_soft_labels, dim=-1)
            view_mrc_loss = F.kl_div(view_prediction_soft_labels, view_mrc_targets, reduction='none').sum(dim=1)
            if obj_prediction_soft_labels is None:
                mrc_loss = view_mrc_loss
            else:
                obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
                obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
                mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
            return mrc_loss.mean()
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_node_types, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            gmap_visited_masks, global_act_labels, local_act_labels, compute_loss
    ):
        batch_size = txt_ids.size(0)
        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_node_types, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )

        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1] != 1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1) - 1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )  # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]  # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j + 1]
                else:
                    tmp[cand_vpid] = local_logits[i, j + 1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        if compute_loss:
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            local_losses = F.cross_entropy(local_logits, local_act_labels, reduction='none')
            fused_losses = F.cross_entropy(fused_logits, global_act_labels, reduction='none')
            losses = global_losses + local_losses + fused_losses
            return losses.mean()
        else:
            return global_logits, local_logits, fused_logits, global_act_labels, local_act_labels

    def forward_rssm(
            self, txt_ids, txt_lens, scans, headings, gt_paths, nav_db, compute_loss, policy_weight, kl_weight, recon_weight, kl_balancing=None
    ):
        batch_size = txt_ids.size(0)
        gmaps = [GraphMap(path[0]) for path in gt_paths]

        txt_masks = gen_seq_masks(txt_lens)
        txt_embeds = self.bert.forward_text(txt_ids, txt_masks)

        past_key_values = None

        neg_selected = set()

        kl_losses = 0
        kl_overshoot_losses = 0
        global_losses = 0
        local_losses = 0
        fused_losses = 0

        unit_cnt = 0
        kl_unit_cnt = 0
        kl_overshoot_unit_cnt = 0

        pred_hidden_batch = []
        pred_kv_batch = []

        pred_rewards_batch = [[] for _ in range(batch_size)]
        gt_rewards_batch = [[] for _ in range(batch_size)]
        pred_img_embeds_batch = [[] for _ in range(batch_size)]
        gt_img_labels_batch = [[] for _ in range(batch_size)]

        all_img_embeds_batch = [[] for _ in range(batch_size)]
        all_rewards_batch = [[] for _ in range(batch_size)]
        all_neg_img_embeds_batch = [[] for _ in range(batch_size)]

        if not compute_loss:
            look_ahead_img_embeds = [[[] for _ in range(batch_size)] for _ in range(5)]
            look_ahead_rewards = [[[] for _ in range(batch_size)] for _ in range(5)]
            gt_look_ahead_indices = [[[] for _ in range(batch_size)] for _ in range(5)]

        pred_gsnn_rewards_batch = [[[] for _ in range(batch_size)] for _ in range(5)]
        gt_gsnn_rewards_batch = [[[] for _ in range(batch_size)] for _ in range(5)]
        pred_gsnn_img_embeds_batch = [[[] for _ in range(batch_size)] for _ in range(5)]
        gt_gsnn_img_labels_batch = [[[] for _ in range(batch_size)] for _ in range(5)]

        trajs = [[path[0]] for path in gt_paths]
        cur_vps = [traj[-1] for traj in trajs]
        cur_headings, cur_elevations, cur_viewidxs = nav_db.get_batch_cur_angles(scans, trajs, headings)
        for t in range(max([len(path) for path in gt_paths])):
            view_img_fts, loc_fts, nav_types, view_lens, cand_vpids = nav_db.get_batch_pano_fts(scans, cur_vps, cur_viewidxs)
            for i, gmap in enumerate(gmaps):
                if t < len(gt_paths[i]):
                    obs = {'viewpoint': cur_vps[i], 'position': nav_db.graphs[scans[i]].nodes[cur_vps[i]]['position']}
                    cands = [{'viewpointId': cand_vp, 'position': nav_db.graphs[scans[i]].nodes[cand_vp]['position']}
                             for cand_vp in cand_vpids[i]]
                    obs['candidate'] = cands
                    gmaps[i].update_graph(obs)
                    gmap.node_step_ids[cur_vps[i]] = t + 1

            if compute_loss:
                noise = self.drop_env(torch.ones(batch_size, self.config.image_feat_size).cuda())
                view_img_fts *= noise.unsqueeze(1).repeat(1, view_img_fts.size(1), 1)
            pano_embeds, pano_masks = self.bert.forward_panorama(view_img_fts, None, loc_fts, nav_types, view_lens, None)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)





            neg_view_img_fts, neg_loc_fts, neg_nav_types, neg_view_lens, neg_cand_vpids, neg_sources, neg_selected = nav_db.get_negative_pano_fts(scans, cur_vps, gt_paths, neg_selected)
            if neg_view_img_fts is not None:
                if compute_loss:
                    noise = self.drop_env(torch.ones(neg_view_img_fts.size(0), self.config.image_feat_size).cuda())
                    neg_view_img_fts *= noise.unsqueeze(1).repeat(1, neg_view_img_fts.size(1), 1)
                neg_pano_embeds, neg_pano_masks = self.bert.forward_panorama(neg_view_img_fts, None, neg_loc_fts, neg_nav_types, neg_view_lens,None)
                neg_avg_pano_embeds = torch.sum(neg_pano_embeds * neg_pano_masks.unsqueeze(2), 1) / torch.sum(neg_pano_masks, 1, keepdim=True)
                for i, source in enumerate(neg_sources):
                    all_neg_img_embeds_batch[source].append(neg_avg_pano_embeds[i])




            if self.bert.recurrent:
                fused_obs_embed = self.bert.state_fusion(torch.cat([avg_pano_embeds, txt_embeds[:, 0, :]], dim=-1))
            else:
                fused_obs_embed = avg_pano_embeds

            if t == 0:
                if 'rssm' in self.bert.world_model_type:
                    deter = self.bert.lang_init(txt_embeds[:, 0, :])
                    post_outs = self.bert.forward_posterior(deter, fused_obs_embed)
                    post_stoch = post_outs['stoch']
                elif 'tssm' in self.bert.world_model_type:
                    deter = self.bert.lang_init(txt_embeds[:, 0, :])
                    post_outs = self.bert.forward_posterior(deter, fused_obs_embed)
                    post_stoch = post_outs['stoch']
                    state_fts = torch.cat([post_outs['stoch'], deter], dim=-1)
                else:
                    raise NotImplementedError
            else:
                if 'rssm' in self.bert.world_model_type:
                    transit_outputs = self.bert.forward_transit(prev_hidden, fused_obs_embed, encoder_embeds=txt_embeds, encoder_attention_mask=txt_masks, past_key_values=past_key_values)
                    state_fts = torch.cat([transit_outputs['posterior_stoch'], transit_outputs['deter']], dim=-1)
                    state_gsnn_fts = torch.cat([transit_outputs['prior_stoch'], transit_outputs['deter']], dim=-1)
                    post_stoch = transit_outputs['posterior_stoch']
                    prior_stoch = transit_outputs['prior_stoch']
                    deter = transit_outputs['deter']
                elif 'tssm' in self.bert.world_model_type:
                    transit_outputs = self.bert.forward_transit(prev_hidden, fused_obs_embed, encoder_embeds=txt_embeds, encoder_attention_mask=txt_masks, past_key_values=past_key_values)
                    state_fts = torch.cat([transit_outputs['posterior_stoch'], transit_outputs['deter']], dim=-1)
                    state_gsnn_fts = torch.cat([transit_outputs['prior_stoch'], transit_outputs['deter']], dim=-1)
                    post_stoch = transit_outputs['posterior_stoch']
                    prior_stoch = transit_outputs['prior_stoch']
                    deter = transit_outputs['deter']
                else:
                    raise NotImplementedError
                past_key_values = transit_outputs['past_key_values']
                recon_outs = self.bert.forward_reconstruct(state_fts)



            for i, gmap in enumerate(gmaps):
                if t < len(gt_paths[i]):
                    gmap.update_node_embed(cur_vps[i], avg_pano_embeds[i], rewrite=True)
                    for j, i_cand_vp in enumerate(cand_vpids[i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            gmap_vpids, gmap_img_embeds, gmap_step_ids, gmap_node_types, gmap_pos_fts, gmap_visited_masks, gmap_pair_dists, gmap_masks, batch_no_vp_left = nav_db.get_batch_gmap_variables(
                cur_vps, cur_headings, cur_elevations, gmaps)
            vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_cand_vpids = nav_db.get_batch_vp_variables(cur_vps,
                                                                                                             cur_headings,
                                                                                                             cur_elevations,
                                                                                                             gmaps,
                                                                                                             pano_embeds,
                                                                                                             cand_vpids,
                                                                                                             view_lens,
                                                                                                             nav_types)
            nav_outputs = self.bert.forward_navigation(txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids,
                                                       gmap_node_types, gmap_pos_fts, gmap_masks, gmap_pair_dists,
                                                       gmap_visited_masks, gmap_vpids, vp_img_embeds, vp_pos_fts,
                                                       vp_masks, vp_nav_masks, None, vp_cand_vpids)
            global_logits, local_logits, fused_logits, state_embeds = nav_outputs['global_logits'], nav_outputs[
                'local_logits'], nav_outputs['fused_logits'], nav_outputs['state_embeds']

            global_act_labels, local_act_labels = [], []
            for i in range(batch_size):
                if t >= len(gt_paths[i]):
                    global_act_label, local_act_label = -100, -100
                else:
                    global_act_label, local_act_label = nav_db.get_act_labels(cur_vps[i], t, gt_paths[i], gmap_vpids[i],
                                                                              [cand_vpids[i]])
                global_act_labels.append(global_act_label)
                local_act_labels.append(local_act_label)

            local_act_labels = torch.LongTensor(local_act_labels).to(txt_ids.device)
            global_act_labels = torch.LongTensor(global_act_labels).to(txt_ids.device)

            global_losses += F.cross_entropy(global_logits, global_act_labels, reduction='none')
            local_losses += F.cross_entropy(local_logits, local_act_labels, reduction='none')
            fused_losses += F.cross_entropy(fused_logits, global_act_labels, reduction='none')

            for i in range(batch_size):
                if t < len(gt_paths[i]) - 1:
                    trajs[i].append(gt_paths[i][t + 1])

            last_vps = copy.deepcopy(cur_vps)
            last_headings = copy.deepcopy(cur_headings)
            last_elevations = copy.deepcopy(cur_elevations)
            cur_vps = [traj[-1] for traj in trajs]
            cur_headings, cur_elevations, cur_viewidxs = nav_db.get_batch_cur_angles(scans, trajs, headings)

            rewards = []
            action_fts = []
            for i, gmap in enumerate(gmaps):
                pos_fts = gmap.get_pos_fts(last_vps[i], [trajs[i][0], cur_vps[i]], last_headings[i], last_elevations[i])
                pos_fts = np.concatenate([pos_fts[0], pos_fts[1]], axis=-1)
                pos_fts = torch.from_numpy(pos_fts).to(txt_ids.device)
                if cur_vps[i] == last_vps[i]:
                    action_fts.append(pos_fts)
                else:
                    action_fts.append(pos_fts)
                rewards.append(nav_db.shortest_distances[scans[i]][cur_vps[i]][gt_paths[i][-1]])

            if self.bert.recurrent:
                action_fts = torch.stack(action_fts, dim=0)
                state_embeds = self.bert.belief_head(torch.cat((state_embeds, action_fts), dim=-1))
                txt_embeds = torch.cat([state_embeds.unsqueeze(1), txt_embeds[:, 1:, :]], 1)

            if 'rssm' in self.bert.world_model_type:
                prev_hidden = [deter, self.bert.stoch_transform(post_stoch)]
                if t > 0:
                    prior_stoch = self.bert.stoch_transform(prior_stoch)
                    prev_gsnn_hidden = [deter, prior_stoch]
            else:
                prev_hidden = state_fts.unsqueeze(1)
                if t > 0:
                    prev_gsnn_hidden = state_gsnn_fts.unsqueeze(1)

            not_ended = torch.tensor([t < len(path) for path in gt_paths], dtype=torch.bool).to(txt_ids.device)
            unit_cnt += not_ended.sum().item()

            # overshooting, gsnn, kl
            if t > 0:
                if t > 1:
                    # overshooting
                    for step_idx, (pred_hidden, prev_past_key_values) in enumerate(zip(pred_hidden_batch, pred_kv_batch)):
                        imagined_outs = self.bert.forward_imagine(pred_hidden, encoder_embeds=txt_embeds, encoder_attention_mask=txt_masks, past_key_values=prev_past_key_values)
                        imagined_state = torch.cat([imagined_outs['stoch'], imagined_outs['deter']], dim=-1)
                        if kl_balancing is None:
                            kl_overshoot_losses += torch.distributions.kl_divergence(transit_outputs['posterior_dist'], imagined_outs['dist']).mean(dim=-1) * not_ended
                        else:
                            imagined_mu_std = imagined_outs['prior_mu_std'].detach()
                            imagined_dist = zdistr(imagined_mu_std, self.bert.stoch_discrete, self.bert.stoch_size)
                            posterior_mu_std = transit_outputs['posterior_mu_std'].detach()
                            posterior_dist = zdistr(posterior_mu_std, self.bert.stoch_discrete, self.bert.stoch_size)
                            kl_overshoot_losses += (1-kl_balancing) * torch.distributions.kl_divergence(transit_outputs['posterior_dist'], imagined_dist).mean(dim=-1) * not_ended
                            kl_overshoot_losses += kl_balancing * torch.distributions.kl_divergence(posterior_dist, imagined_outs['dist']).mean(dim=-1) * not_ended
                        kl_overshoot_unit_cnt += not_ended.sum().item()

                        # gsnn
                        imagined_recon_outs = self.bert.forward_reconstruct(imagined_state)
                        for i in range(batch_size):
                            if t < len(gt_paths[i]):
                                look_ahead_step = t - step_idx - 1
                                assert look_ahead_step > 0
                                if look_ahead_step < 5:
                                    pred_gsnn_img_embeds_batch[look_ahead_step][i].append(imagined_recon_outs['img_embed'][i])
                                    pred_gsnn_rewards_batch[look_ahead_step][i].append(imagined_recon_outs['reward'][i])
                                    # gt_gsnn_img_embeds_batch[look_ahead_step][i].append(avg_pano_embeds[i].detach())
                                    gt_gsnn_img_labels_batch[look_ahead_step][i].append(t)
                                    gt_gsnn_rewards_batch[look_ahead_step][i].append(rewards[i])
                                    if not compute_loss:
                                        look_ahead_img_embeds[look_ahead_step][i].append(imagined_recon_outs['img_embed'][i])
                                        look_ahead_rewards[look_ahead_step][i].append(imagined_recon_outs['reward'][i])
                                        gt_look_ahead_indices[look_ahead_step][i].append(t)

                        if 'rssm' in self.bert.world_model_type:
                            pred_hidden_batch[step_idx] = [imagined_outs['deter'], self.bert.stoch_transform(imagined_outs['stoch'])]
                        elif 'tssm' in self.bert.world_model_type:
                            pred_hidden_batch[step_idx] = imagined_state.unsqueeze(1)
                        else:
                            raise NotImplementedError
                        pred_kv_batch[step_idx] = imagined_outs['past_key_values']

                # gsnn
                recon_gsnn_outs = self.bert.forward_reconstruct(state_gsnn_fts)
                for i in range(batch_size):
                    if t < len(gt_paths[i]):
                        pred_gsnn_img_embeds_batch[0][i].append(recon_gsnn_outs['img_embed'][i])
                        pred_gsnn_rewards_batch[0][i].append(recon_gsnn_outs['reward'][i])
                        gt_gsnn_img_labels_batch[0][i].append(t)
                        gt_gsnn_rewards_batch[0][i].append(rewards[i])

                        if not compute_loss:
                            look_ahead_img_embeds[0][i].append(recon_gsnn_outs['img_embed'][i])
                            look_ahead_rewards[0][i].append(recon_gsnn_outs['reward'][i])
                            gt_look_ahead_indices[0][i].append(t)

                # kl
                if kl_balancing is None:
                    kl_losses += torch.distributions.kl_divergence(transit_outputs['posterior_dist'], transit_outputs['prior_dist']).mean(dim=-1) * not_ended
                else:
                    prior_mu_std = transit_outputs['prior_mu_std'].detach()
                    prior_dist = zdistr(prior_mu_std, self.bert.stoch_discrete, self.bert.stoch_size)
                    posterior_mu_std = transit_outputs['posterior_mu_std'].detach()
                    posterior_dist = zdistr(posterior_mu_std, self.bert.stoch_discrete, self.bert.stoch_size)
                    kl_losses += (1-kl_balancing) * torch.distributions.kl_divergence(transit_outputs['posterior_dist'], prior_dist).mean(dim=-1) * not_ended
                    kl_losses += kl_balancing * torch.distributions.kl_divergence(posterior_dist, transit_outputs['prior_dist']).mean(dim=-1) * not_ended
                kl_unit_cnt += not_ended.sum().item()

                pred_hidden_batch.append(prev_gsnn_hidden)
                pred_kv_batch.append(past_key_values)

                for i in range(batch_size):
                    if t < len(gt_paths[i]):
                        pred_rewards_batch[i].append(recon_outs['reward'][i])
                        gt_rewards_batch[i].append(rewards[i])
                        pred_img_embeds_batch[i].append(recon_outs['img_embed'][i])
                        gt_img_labels_batch[i].append(t)

            for i in range(batch_size):
                if t < len(gt_paths[i]):
                    all_img_embeds_batch[i].append(avg_pano_embeds[i])
                    all_rewards_batch[i].append(rewards[i])

        full_all_img_embeds, full_pred_img_embeds, full_gt_img_labels, full_pred_rewards, full_gt_rewards = [], [], [], [], []
        full_pred_gsnn_img_embeds = [[] for _ in range(5)]
        full_gt_gsnn_img_labels = [[] for _ in range(5)]
        full_pred_gsnn_rewards = [[] for _ in range(5)]
        full_gt_gsnn_rewards = [[] for _ in range(5)]
        cum_length = 0
        for i in range(batch_size):
            full_all_img_embeds.extend(all_img_embeds_batch[i])
            full_pred_img_embeds.extend(pred_img_embeds_batch[i])
            full_gt_img_labels.extend([index+cum_length for index in gt_img_labels_batch[i]])
            full_pred_rewards.extend(pred_rewards_batch[i])
            full_gt_rewards.extend(gt_rewards_batch[i])
            for j in range(5):
                full_pred_gsnn_img_embeds[j].extend(pred_gsnn_img_embeds_batch[j][i])
                full_gt_gsnn_img_labels[j].extend([index+cum_length for index in gt_gsnn_img_labels_batch[j][i]])
                full_pred_gsnn_rewards[j].extend(pred_gsnn_rewards_batch[j][i])
                full_gt_gsnn_rewards[j].extend(gt_gsnn_rewards_batch[j][i])
            cum_length += len(all_img_embeds_batch[i])
        full_all_img_embeds = torch.stack(full_all_img_embeds, dim=0)
        full_pred_img_embeds = torch.stack(full_pred_img_embeds, dim=0)
        full_gt_img_labels = torch.LongTensor(full_gt_img_labels).to(txt_ids.device)
        full_pred_rewards = torch.stack(full_pred_rewards, dim=0)
        full_gt_rewards = torch.FloatTensor(full_gt_rewards).to(txt_ids.device)

        full_pred_gsnn_img_embeds = [torch.stack(full_pred_gsnn_img_embeds[j], dim=0) for j in range(5)]
        full_gt_gsnn_img_labels = [torch.LongTensor(full_gt_gsnn_img_labels[j]).to(txt_ids.device) for j in range(5)]
        full_pred_gsnn_rewards = [torch.stack(full_pred_gsnn_rewards[j], dim=0) for j in range(5)]
        full_gt_gsnn_rewards = [torch.FloatTensor(full_gt_gsnn_rewards[j]).to(txt_ids.device) for j in range(5)]

        all_neg_img_embeds = []
        all_img_embed_lengths = []
        for i in range(batch_size):
            all_img_embed_lengths.append(len(all_img_embeds_batch[i]))
            all_neg_img_embeds.extend(all_neg_img_embeds_batch[i])
        all_neg_img_embeds = torch.stack(all_neg_img_embeds, dim=0)

        full_all_img_embeds = torch.cat([full_all_img_embeds, all_neg_img_embeds], dim=0)
        full_gsnn_pred_img_embeds = torch.cat(full_pred_gsnn_img_embeds, dim=0)
        full_gsnn_gt_img_labels = torch.cat(full_gt_gsnn_img_labels, dim=0)
        full_gsnn_pred_rewards = torch.cat(full_pred_gsnn_rewards, dim=0)
        full_gsnn_gt_rewards = torch.cat(full_gt_gsnn_rewards, dim=0)
        
        # loss
        full_all_img_embeds = self.bert.obs_transform(full_all_img_embeds)

        recon_act_losses = F.mse_loss(full_pred_rewards.view(-1), full_gt_rewards, reduction="mean")
        recon_vis_losses = info_nce_loss(full_pred_img_embeds, full_all_img_embeds, full_gt_img_labels, reduction="mean")
        recon_gsnn_act_losses = F.mse_loss(full_gsnn_pred_rewards.view(-1), full_gsnn_gt_rewards, reduction="mean")
        recon_gsnn_vis_losses = info_nce_loss(full_gsnn_pred_img_embeds, full_all_img_embeds, full_gsnn_gt_img_labels, reduction="mean")

        kl_overshoot_losses = kl_overshoot_losses.sum() / kl_overshoot_unit_cnt
        kl_losses = kl_losses.sum() / kl_unit_cnt
        # recon_act_losses = F.mse_loss(full_pred_rewards.view(-1), full_gt_rewards, reduction="sum") / batch_size
        # recon_vis_losses = info_nce_loss(full_pred_img_embeds, full_all_img_embeds, full_gt_img_labels, reduction="sum") / batch_size
        # recon_gsnn_act_losses = F.mse_loss(full_gsnn_pred_rewards.view(-1), full_gsnn_gt_rewards, reduction="sum") / (batch_size * 5)
        # recon_gsnn_vis_losses = info_nce_loss(full_gsnn_pred_img_embeds, full_all_img_embeds, full_gsnn_gt_img_labels, reduction="sum") / (batch_size * 5)
        #
        # kl_overshoot_losses = kl_overshoot_losses.sum() / (batch_size * 5)
        # kl_losses = kl_losses.sum() / batch_size

        global_losses = global_losses.sum() / unit_cnt
        local_losses = local_losses.sum() / unit_cnt
        fused_losses = fused_losses.sum() / unit_cnt

        # metrics
        final_reward_preds = full_pred_rewards.view(-1)
        final_rewards = full_gt_rewards

        final_img_preds = []
        final_hard_img_preds = []
        final_img_labels = []
        final_img_soft_labels = []
        for i in range(batch_size):
            all_img_embeds_batch[i] = torch.stack(all_img_embeds_batch[i], dim=0)
            all_neg_img_embeds_batch[i] = torch.stack(all_neg_img_embeds_batch[i], dim=0) if len(all_neg_img_embeds_batch[i]) > 0 else None
            if all_neg_img_embeds_batch[i] is not None:
                all_img_embeds_batch[i] = torch.cat([all_img_embeds_batch[i], all_neg_img_embeds_batch[i]], dim=0)
            all_img_embeds_batch[i] = self.bert.obs_transform(all_img_embeds_batch[i])
            img_logits = torch.matmul(torch.stack(pred_img_embeds_batch[i]), all_img_embeds_batch[i].t())
            final_img_preds.append(torch.argmax(img_logits[:,:all_img_embed_lengths[i]], dim=-1))
            final_hard_img_preds.append(torch.argmax(img_logits, dim=-1))
            final_img_labels.append(torch.LongTensor(range(1, all_img_embed_lengths[i])).to(txt_ids.device))
            final_img_soft_labels.append(torch.LongTensor([all_img_embed_lengths[i]] * len(pred_img_embeds_batch[i])).to(txt_ids.device))
        final_img_preds = torch.cat(final_img_preds, dim=0)
        final_hard_img_preds = torch.cat(final_hard_img_preds, dim=0)
        final_img_labels = torch.cat(final_img_labels, dim=0)
        final_img_soft_labels = torch.cat(final_img_soft_labels, dim=0)

        # imagine metrics
        if not compute_loss:
            final_look_ahead_reward_preds = [[] for _ in range(5)]
            final_look_ahead_rewards = [[] for _ in range(5)]
            final_look_ahead_img_preds = [[] for _ in range(5)]
            final_look_ahead_hard_img_preds = [[] for _ in range(5)]
            final_look_ahead_img_labels = [[] for _ in range(5)]
            final_look_ahead_img_soft_labels = [[] for _ in range(5)]
            for i in range(5):
                for j in range(batch_size):
                    if len(look_ahead_img_embeds[i][j]) > 0:
                        look_ahead_rewards_tmp = torch.stack(look_ahead_rewards[i][j], dim=0)
                        rewards_tmp = look_ahead_rewards_tmp.view(-1)
                        final_look_ahead_reward_preds[i].append(rewards_tmp)
                        rewards = [all_rewards_batch[j][idx] for idx in gt_look_ahead_indices[i][j]]
                        final_look_ahead_rewards[i].append(torch.FloatTensor(rewards).to(txt_ids.device))

                        img_logits = torch.matmul(torch.stack(look_ahead_img_embeds[i][j]), all_img_embeds_batch[j].t())
                        final_look_ahead_img_preds[i].append(torch.argmax(img_logits[:,:all_img_embed_lengths[j]], dim=-1))
                        final_look_ahead_hard_img_preds[i].append(torch.argmax(img_logits, dim=-1))
                        final_look_ahead_img_labels[i].append(torch.LongTensor(gt_look_ahead_indices[i][j]).to(txt_ids.device))
                        final_look_ahead_img_soft_labels[i].append(torch.LongTensor([all_img_embed_lengths[j]] * len(gt_look_ahead_indices[i][j])).to(txt_ids.device))

                final_look_ahead_reward_preds[i] = torch.cat(final_look_ahead_reward_preds[i], dim=0)
                final_look_ahead_rewards[i] = torch.cat(final_look_ahead_rewards[i], dim=0)
                final_look_ahead_img_preds[i] = torch.cat(final_look_ahead_img_preds[i], dim=0)
                final_look_ahead_hard_img_preds[i] = torch.cat(final_look_ahead_hard_img_preds[i], dim=0)
                final_look_ahead_img_labels[i] = torch.cat(final_look_ahead_img_labels[i], dim=0)
                final_look_ahead_img_soft_labels[i] = torch.cat(final_look_ahead_img_soft_labels[i], dim=0)

        if compute_loss:
            return policy_weight * (
                        global_losses + local_losses + fused_losses) + kl_weight * kl_losses + kl_weight * kl_overshoot_losses + recon_weight * (0.1*recon_act_losses + 0.1*recon_gsnn_act_losses + recon_vis_losses + recon_gsnn_vis_losses)
        else:
            return global_losses, local_losses, fused_losses, kl_losses, recon_vis_losses, recon_act_losses, final_reward_preds, final_rewards, final_img_preds, final_hard_img_preds, final_img_labels, final_img_soft_labels, final_look_ahead_reward_preds, final_look_ahead_rewards, final_look_ahead_img_preds, final_look_ahead_hard_img_preds, final_look_ahead_img_labels, final_look_ahead_img_soft_labels
