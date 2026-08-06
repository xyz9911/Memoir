import copy
import sys
from collections import defaultdict

import numpy as np
import math

import torch

from nav_src.envs.eval_utils import cal_dtw

from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F

from nav_src.gsa_nav.agent_base import Seq2SeqAgent

from utils.graph_utils import CollectiveGraphMap, get_historical_experience_prompt
from vlnbert.model import VLNBert
from utils.ops import pad_tensors_wgrad, info_nce_loss, zdistr


class GMapNavAgent(Seq2SeqAgent):

    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.scanvp_cands = {}
        self.scan_cnt = defaultdict(int)
        self.scan_gmap_bank = {}
        self.env_max_traj_num = {}

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        batch_view_ids = []

        for i, ob in enumerate(obs):
            self.store_obs["%s_%s" % (ob['scan'], ob['viewpoint'])] = ob
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = []
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.append(cc['pointId'])
            batch_view_ids.append(used_viewidxs)
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(set(used_viewidxs))))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            'cand_vpids': batch_cand_vpids, 'view_ids': batch_view_ids
        }

    def _nav_gmap_variable(self, obs, gmaps, gmap_id_map, robot_id_map):
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks, batch_gmap_teacher_masks = [], [], []
        batch_gmap_node_types = []
        batch_no_vp_left = []
        for i, ob in enumerate(obs):
            gmap_id = gmap_id_map[i]
            robot_id = robot_id_map[i]
            visited_vpids, prompted_unreachable_vpids, prompted_reachable_vpids, unvisited_unreachable_vpids, unvisited_reachable_vpids = [], [], [], [], []
            prompted_vpids, unvisited_vpids = [], []
            if self.args.forest:
                connected_nodes, other_nodes = gmaps[gmap_id].get_connected_nodes(robot_id, full_graph=True)
                local_graph_nodes = gmaps[gmap_id].get_connected_nodes(robot_id)
                visited_vpids += other_nodes
                for k in connected_nodes:
                    if k in local_graph_nodes:
                        if gmaps[gmap_id].visited(k, robot_id):
                            visited_vpids.append(k)
                        elif gmaps[gmap_id].observed(k, robot_id):
                            if gmaps[gmap_id].reachable(k, robot_id):
                                prompted_reachable_vpids.append(k)
                            else:
                                prompted_unreachable_vpids.append(k)
                            prompted_vpids.append(k)
                        else:
                            if gmaps[gmap_id].reachable(k, robot_id):
                                unvisited_reachable_vpids.append(k)
                            else:
                                unvisited_unreachable_vpids.append(k)
                            unvisited_vpids.append(k)
                    else:
                        visited_vpids.append(k)
            else:
                local_graph_nodes = gmaps[gmap_id].get_connected_nodes(robot_id)
                for k in local_graph_nodes:
                    if gmaps[gmap_id].visited(k, robot_id):
                        visited_vpids.append(k)
                    elif gmaps[gmap_id].observed(k, robot_id):
                        if gmaps[gmap_id].reachable(k, robot_id):
                            prompted_reachable_vpids.append(k)
                        else:
                            prompted_unreachable_vpids.append(k)
                        prompted_vpids.append(k)
                    else:
                        if gmaps[gmap_id].reachable(k, robot_id):
                            unvisited_reachable_vpids.append(k)
                        else:
                            unvisited_unreachable_vpids.append(k)
                        unvisited_vpids.append(k)

            batch_no_vp_left.append(len(prompted_reachable_vpids + unvisited_reachable_vpids) == 0)
            if self.args.teacher_teleport:
                gmap_vpids = [None] + visited_vpids + prompted_vpids + unvisited_vpids
                gmap_teacher_masks = [0] + [1] * len(visited_vpids) + [0] * len(prompted_vpids + unvisited_vpids)
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(prompted_vpids + unvisited_vpids)
                gmap_node_types = [0] + [1] * len(visited_vpids) + [2] * len(prompted_vpids) + [3] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + visited_vpids + prompted_unreachable_vpids + unvisited_unreachable_vpids + prompted_reachable_vpids + unvisited_reachable_vpids
                gmap_teacher_masks = [0] + [1] * len(visited_vpids + prompted_unreachable_vpids + unvisited_unreachable_vpids) + [0] * len(prompted_reachable_vpids + unvisited_reachable_vpids)
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(prompted_unreachable_vpids + unvisited_unreachable_vpids + prompted_reachable_vpids + unvisited_reachable_vpids)
                gmap_node_types = [0] + [1] * len(visited_vpids) + [2] * len(prompted_unreachable_vpids) + [3] * len(unvisited_unreachable_vpids) + [2] * len(prompted_reachable_vpids) + [3] * len(unvisited_reachable_vpids)

            gmap_step_ids = [gmaps[gmap_id].node_step_ids[robot_id].get(vp, 0) for vp in gmap_vpids]
            if len(gmap_vpids) != 1:
                gmap_img_embeds = [gmaps[gmap_id].get_node_embed(vp, robot_id, self.args.forest) for vp in gmap_vpids[1:]]
                gmap_img_embeds = torch.stack(
                    [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
                )  # cuda
            else:
                gmap_img_embeds = torch.stack(
                    [torch.zeros(self.args.image_feat_size, device=self.device)], 0
                )

            gmap_pos_fts = gmaps[gmap_id].get_pos_fts(
                ob['viewpoint'], gmap_vpids, ob['heading'], ob['elevation'], robot_id
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for u in range(1, len(gmap_vpids)):
                for v in range(u + 1, len(gmap_vpids)):
                    gmap_pair_dists[u, v] = gmap_pair_dists[v, u] = \
                        gmaps[gmap_id].distance(gmap_vpids[u], gmap_vpids[v], robot_id)

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_node_types.append(torch.LongTensor(gmap_node_types))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_teacher_masks.append(torch.BoolTensor(gmap_teacher_masks))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_node_types = pad_sequence(batch_gmap_node_types, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_teacher_masks = pad_sequence(batch_gmap_teacher_masks, batch_first=True).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_node_types': batch_gmap_node_types,
            'gmap_pos_fts': batch_gmap_pos_fts, 'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }, batch_gmap_teacher_masks

    def update_emap_nodes(self, emap_nodes, exp_pairs, exp_sims, exp_traj_ids):
        for emap_node_set, exp_pair, exp_sim, exp_traj_id in zip(emap_nodes, exp_pairs, exp_sims, exp_traj_ids):
            for pair, sim, traj_id in zip(exp_pair, exp_sim, exp_traj_id):
                for j, (node, fts) in enumerate(pair):
                    if fts is not None:
                        if node not in emap_node_set:
                            emap_node_set[node] = [[fts[0], sim[j], traj_id]]
                        else:
                            emap_node_set[node].append([fts[0], sim[j], traj_id])
        return emap_nodes

    def _nav_emap_variable(self, obs, gmaps, state_fts, emap_nodes, gmap_id_map, robot_id_map, multimodal_history):
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks, batch_gmap_teacher_masks = [], [], []
        batch_gmap_node_types = []

        all_state_fts = []
        all_memory_fts = []
        all_similarities = []
        all_seq_lengths = []
        batch_nodes = [[] for _ in obs]
        batch_emap_embeds = [[] for _ in obs]
        for i, emap_node_set in enumerate(emap_nodes):
            cnt = 0
            for node, embed_items in emap_node_set.items():
                embeds = []
                sims = []
                for embed_item in embed_items:
                    embeds.append(embed_item[0])
                    sims.append(embed_item[1])
                all_state_fts.append(state_fts[i])
                all_memory_fts.append(torch.stack(embeds))
                if None not in sims:
                    all_similarities.append(torch.stack(sims))
                else:
                    all_similarities.append(None)
                batch_nodes[i].append(node)
                all_seq_lengths.append(len(embeds))
                cnt += 1

        if len(all_state_fts) and multimodal_history:
            if self.args.exp_fusion == "attention":
                all_state_fts = torch.stack(all_state_fts).unsqueeze(1)
                all_state_fts = self.vln_bert.vln_bert.exp_transform(all_state_fts)
                all_memory_fts = pad_sequence(all_memory_fts, batch_first=True)
                all_memory_fts = self.vln_bert.vln_bert.exp_transform(all_memory_fts)

                max_seq_length = max(all_seq_lengths)
                attention_mask = torch.zeros((len(all_seq_lengths), max_seq_length), dtype=torch.bool).cuda()
                for i, length in enumerate(all_seq_lengths):
                    attention_mask[i, :length] = -float('inf')
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                memory_fts = self.vln_bert("memory", {"state_fts": all_state_fts, "memory_fts": all_memory_fts, "memory_attention_mask": attention_mask})[0]
            elif self.args.exp_fusion == "similarity":
                all_memory_fts = pad_sequence(all_memory_fts, batch_first=True)
                all_similarities = pad_sequence(all_similarities, batch_first=True, padding_value=-float('inf'))
                all_similarities = F.softmax(all_similarities / 0.1, dim=-1)
                memory_fts = torch.sum(all_memory_fts * all_similarities.unsqueeze(-1), 1)
                memory_fts = self.vln_bert.vln_bert.exp_transform(memory_fts)
            elif self.args.exp_fusion == "mean":
                memory_fts = [fts.mean(0) for fts in all_memory_fts]
                memory_fts = torch.stack(memory_fts)
                memory_fts = self.vln_bert.vln_bert.exp_transform(memory_fts)

        cnt = 0
        for i, nodes in enumerate(batch_nodes):
            gmap_id = gmap_id_map[i]
            robot_id = robot_id_map[i]
            if len(nodes):
                if multimodal_history:
                    batch_emap_embeds[i] = [gmaps[gmap_id].get_node_embed(node, robot_id, self.args.forest) + memory_ft for node, memory_ft in zip(nodes, memory_fts[cnt:cnt+len(nodes)])]
                else:
                    batch_emap_embeds[i] = [gmaps[gmap_id].get_node_embed(node, robot_id, self.args.forest) for node in nodes]
                cnt += len(nodes)

        for i, ob in enumerate(obs):
            gmap_id = gmap_id_map[i]
            robot_id = robot_id_map[i]
            visited_vpids, exp_unreachable_vpids, exp_reachable_vpids = [], [], []
            exp_vpids = []
            exp_embeds, exp_embed_cnts = [], []

            if self.args.forest:
                connected_nodes, other_nodes = gmaps[gmap_id].get_connected_nodes(robot_id, full_graph=True)
                visited_vpids += other_nodes
                for k in connected_nodes:
                    if gmaps[gmap_id].visited(k, robot_id) or k not in batch_nodes[i]:
                        visited_vpids.append(k)
            else:
                local_graph_nodes = gmaps[gmap_id].get_connected_nodes(robot_id)
                for k in local_graph_nodes:
                    if gmaps[gmap_id].visited(k, robot_id):
                        visited_vpids.append(k)
            for node, emap_embed in zip(batch_nodes[i], batch_emap_embeds[i]):
                if node not in visited_vpids and node != ob['viewpoint']:
                    assert node not in exp_unreachable_vpids and node not in exp_reachable_vpids
                    if gmaps[gmap_id].reachable(node, robot_id):
                        exp_reachable_vpids.append(node)
                    else:
                        exp_unreachable_vpids.append(node)
                    exp_embeds.append(emap_embed.squeeze(0))
                    exp_vpids.append(node)

            if self.args.teacher_teleport:
                gmap_vpids = [None] + visited_vpids + exp_vpids
                gmap_teacher_masks = [0] + [1] * len(visited_vpids) + [0] * len(exp_vpids)
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(exp_vpids)
                gmap_node_types = [0] + [1] * len(visited_vpids) + [2] * len(exp_vpids)
            else:
                gmap_vpids = [None] + visited_vpids + exp_unreachable_vpids + exp_reachable_vpids
                gmap_teacher_masks = [0] + [1] * len(visited_vpids + exp_unreachable_vpids) + [0] * len(exp_reachable_vpids)
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(exp_unreachable_vpids + exp_reachable_vpids)
                gmap_node_types = [0] + [1] * len(visited_vpids) + [2] * len(exp_unreachable_vpids + exp_reachable_vpids)

            gmap_step_ids = [gmaps[gmap_id].node_step_ids[robot_id].get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmaps[gmap_id].get_node_embed(vp, robot_id, self.args.forest) for vp in visited_vpids]

            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds + exp_embeds, 0
            )

            gmap_pos_fts = gmaps[gmap_id].get_pos_fts(
                ob['viewpoint'], gmap_vpids, ob['heading'], ob['elevation'], robot_id
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for u in range(1, len(gmap_vpids)):
                for v in range(u + 1, len(gmap_vpids)):
                    gmap_pair_dists[u, v] = gmap_pair_dists[v, u] = \
                        gmaps[gmap_id].distance(gmap_vpids[u], gmap_vpids[v], robot_id)

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_node_types.append(torch.LongTensor(gmap_node_types))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_teacher_masks.append(torch.BoolTensor(gmap_teacher_masks))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_node_types = pad_sequence(batch_gmap_node_types, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_teacher_masks = pad_sequence(batch_gmap_teacher_masks, batch_first=True).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'emap_vpids': batch_gmap_vpids, 'emap_img_embeds': batch_gmap_img_embeds,
            'emap_step_ids': batch_gmap_step_ids, 'emap_node_types': batch_gmap_node_types,
            'emap_pos_fts': batch_gmap_pos_fts, 'emap_visited_masks': batch_gmap_visited_masks,
            'emap_pair_dists': gmap_pair_dists, 'emap_masks': batch_gmap_masks
        }, batch_gmap_teacher_masks

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types, gmap_id_map, robot_id_map):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, ob in enumerate(obs):
            gmap_id = gmap_id_map[i]
            robot_id = robot_id_map[i]
            cur_cand_pos_fts = gmaps[gmap_id].get_pos_fts(
                ob['viewpoint'], cand_vpids[i],
                ob['heading'], ob['elevation'], robot_id
            )
            cur_start_pos_fts = gmaps[gmap_id].get_pos_fts(
                ob['viewpoint'], [gmaps[gmap_id].start_vps[robot_id]],
                ob['heading'], ob['elevation'], robot_id
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()
        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens + 1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None] + x for x in cand_vpids],
        }

    def _teacher_path(self, obs, gmaps, gmap_id_map, robot_id_map, ended):
        """
        Extract teacher paths into variable.
        :param obs: The observation
        :param vpids: available nodes for searching
        :param ended: Whether the action seq is ended
        :param visited_masks: Optional mask for visited nodes
        :return: Tuple of (paths, stop_flags) where paths contains sequences of viewpoints
                and stop_flags indicates if the path should terminate
        """
        paths = []
        stop_flags = []

        for i, ob in enumerate(obs):
            gmap = gmaps[gmap_id_map[i]]
            if ended[i]:  # Just ignore this index
                path = None
                stop_flag = None
            else:
                vpids = gmap.get_available_nodes(ob['viewpoint'], robot_id_map[i], look_ahead_steps=self.args.look_ahead_steps)
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    # Already at goal
                    path = []
                    stop_flag = True
                elif ob['viewpoint'] in ob['gt_path']:
                    # Currently on the ground truth path
                    cur_vp = ob['viewpoint']
                    cur_idx = ob['gt_path'].index(cur_vp)
                    path = []
                    best_idx = cur_idx

                    # Try to find path through available viewpoints
                    for j, vpid in enumerate(vpids):
                        # Check if this viewpoint helps us get closer to target
                        if vpid in ob['gt_path'][cur_idx:]:
                            # This viewpoint is on the remaining ground truth path
                            next_vp_idx = ob['gt_path'].index(vpid)
                            if next_vp_idx > best_idx:  # Ensures forward progress
                                path = gmap.graph_pool.path(cur_vp, vpid)
                                best_idx = next_vp_idx

                    # If no direct path found, try to find closest point to target
                    if best_idx == cur_idx:
                        path = []
                    stop_flag = False
                else:
                    # Not on ground truth path, prioritize getting closer to target
                    cur_vp = ob['viewpoint']
                    scan = ob['scan']
                    target_vp = ob['gt_path'][-1]
                    min_dist = self.env.shortest_distances[scan][cur_vp][target_vp]
                    min_bridge_dist = self.env.shortest_distances[scan][cur_vp][target_vp]
                    max_in_gt_path = False
                    best_vpid = ob['viewpoint']

                    # Find viewpoint that gets us closest to target
                    for j, vpid in enumerate(vpids):
                        dist = gmap.graph_pool.distance(cur_vp, vpid) + self.env.shortest_distances[scan][vpid][target_vp]
                        bridge_dist = self.env.shortest_distances[scan][vpid][target_vp]
                        in_gt_path = vpid in ob['gt_path']
                        if abs(dist - min_dist) < 0.01:
                            if in_gt_path > max_in_gt_path:
                                min_dist = dist
                                min_bridge_dist = bridge_dist
                                best_vpid = vpid
                                max_in_gt_path = in_gt_path
                            elif in_gt_path == max_in_gt_path and bridge_dist < min_bridge_dist:
                                min_dist = dist
                                min_bridge_dist = bridge_dist
                                best_vpid = vpid
                                max_in_gt_path = in_gt_path

                    if best_vpid != ob['viewpoint'] and min_dist < 15:
                        path = gmap.graph_pool.path(cur_vp, best_vpid)
                    else:
                        path = []
                    stop_flag = False

            paths.append(path)
            stop_flags.append(stop_flag)

        return paths, stop_flags

    def _teacher_action_r4r(
            self, obs, vpids, emap_vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        action_names = []
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
                action_name = None
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0  # stop
                        action_name = ob['viewpoint']
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                action_name = vpid
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0  # Stop if arrived
                        action_name = ob['viewpoint']
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        min_bridge_dist = float('inf')
                        max_in_emap = False
                        min_idxes = []
                        min_action_names = []
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan],
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][
                                                                   1:],
                                        ob['gt_path'],
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                           + self.env.shortest_distances[scan][cur_vp][vpid]
                                    bridge_dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]]
                                    in_emap = vpid in emap_vpids[i]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                                    min_bridge_dist = bridge_dist
                                    max_in_emap = in_emap
                                    action_name = vpid
                                    min_idxes = [min_idx]
                                    min_action_names = [action_name]
                                elif dist == min_dist:
                                    min_idxes.append(j)
                                    min_action_names.append(vpid)
                                    if self.args.precise_teleport:
                                        if in_emap > max_in_emap:
                                            min_dist = dist
                                            min_idx = j
                                            min_bridge_dist = bridge_dist
                                            max_in_emap = in_emap
                                            action_name = vpid
                                        elif in_emap == max_in_emap and bridge_dist < min_bridge_dist:
                                            min_dist = dist
                                            min_idx = j
                                            min_bridge_dist = bridge_dist
                                            max_in_emap = in_emap
                                            action_name = vpid
                        if self.args.teacher_aug and min_idx != self.args.ignoreid:
                            assert len(min_idxes) == len(min_action_names)
                            idx = np.random.randint(len(min_idxes))
                            min_idx = min_idxes[idx]
                            action_name = min_action_names[idx]
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
                            action_name = None
            action_names.append(action_name)
        return torch.from_numpy(a).cuda(), action_names

    def make_equiv_action(self, gmaps, a_t, raw_obs, t, robot_id_map, traj):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        cnt = 0
        tot_cnt = 0
        for i, raw_obs_cluster in enumerate(raw_obs):
            for j, ob in enumerate(raw_obs_cluster):
                if ob is not None:
                    action = a_t[cnt]
                    if action is not None:  # None is the <stop> action
                        traj[cnt]['path'].append(
                            gmaps[tot_cnt].graph_pool.path(ob['viewpoint'], action))
                        if len(traj[cnt]['path'][-1]) == 1:
                            prev_vp = traj[cnt]['path'][-2][-1]
                        else:
                            prev_vp = traj[cnt]['path'][-1][-2]
                        viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                        heading = (viewidx % 12) * math.radians(30)
                        elevation = (viewidx // 12 - 1) * math.radians(30)
                        self.env.env.sims[i][j].newEpisode([ob['scan']], [action], [heading], [elevation])
                    cnt += 1
                tot_cnt += 1
        return self.env._get_obs(t=t)

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    def rollout(self, train_ml=None, reset=True, extended_memory=True):
        raw_obs, tour_start = self.env.reset()

        obs = []
        robot_id_map = {}
        gmap_id_map = {}
        cluster_ix = {}
        cnt = 0
        tot_cnt = 0
        for i, cluster_obs in enumerate(raw_obs):
            valid_obs = []
            for j, ob in enumerate(cluster_obs):
                if ob is not None:
                    valid_obs.append(ob)
                    robot_id_map[cnt] = j
                    gmap_id_map[cnt] = tot_cnt
                    cluster_ix[cnt] = i
                    cnt += 1
                tot_cnt += 1
            obs.extend(valid_obs)
        batch_size = cnt
        tour_not_ended = ~self.env.tour_ended
        assert batch_size == tour_not_ended.sum()

        self._update_scanvp_cands(obs)

        for scan, count in self.scan_cnt.items():
            if train_ml is not None and count > self.env_max_traj_num[scan]:
                self.scan_gmap_bank.pop(scan)
                self.scan_cnt[scan] = 0

        gmaps = [None for _ in raw_obs]
        gmap_analysis = defaultdict(list)
        cnt = 0
        for i, cluster_obs in enumerate(raw_obs):
            if None not in cluster_obs:
                scan = cluster_obs[0]['scan']
                if scan not in self.scan_gmap_bank:
                    if self.args.share_graph:
                        gmap = CollectiveGraphMap([ob['viewpoint'] for ob in cluster_obs], shared=True)
                    else:
                        gmap = CollectiveGraphMap([ob['viewpoint'] for ob in cluster_obs])
                    gmaps[cnt] = gmap
                    self.scan_gmap_bank[cluster_obs[0]['scan']] = gmap
                    self.scan_cnt[scan] = 0
                else:
                    if len(gmap_analysis[scan]) == 0:
                        gmaps[cnt] = self.scan_gmap_bank[scan]
                        start_vps = [ob['viewpoint'] for ob in cluster_obs]
                        gmaps[cnt].reset(start_vps, clear=not extended_memory)
                    else:
                        start_vps = [ob['viewpoint'] for ob in cluster_obs]
                        gmaps[cnt] = self.scan_gmap_bank[scan].create_episode_view(start_vps)
                gmap_analysis[scan].append(cnt)
            cnt += len(cluster_obs)

        # build graph: keep the start viewpoint
        for i, ob in enumerate(obs):
            self.scan_cnt[ob['scan']] += 1
            gmap_id = gmap_id_map[i]
            gmaps[gmap_id].update_graph(ob, robot_id_map[i])

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'scan': ob['scan'],
            'cluster_id': cluster_ix[i],
            'robot_id': robot_id_map[i],
            'gt_path': ob['gt_path'],
            'gt_length': ob['distance'],
            'path': [[ob['viewpoint']]],
            'imagined_obs': [[] for _ in range(5)],
            'obs_labels': [[] for _ in range(5)],
            'obs_soft_labels': [[] for _ in range(5)],
            'imagined_actions': [[] for _ in range(5)],
            'action_labels': [[] for _ in range(5)],
            'details': {}
        } for i, ob in enumerate(obs)]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)
        txt_masks = language_inputs['txt_masks']

        # Initialization the tracking state
        ended = torch.BoolTensor([False for _ in range(batch_size)]).to(txt_embeds.device)
        just_ended = torch.BoolTensor([False for _ in range(batch_size)]).to(txt_embeds.device)

        # Init the logs
        entropys = []

        ml_losses = 0.
        kl_losses = 0.
        kl_overshoot_losses = 0.
        recon_reward_losses = 0.
        recon_vis_losses = 0.
        recon_reward_gsnn_losses = 0.
        recon_vis_gsnn_losses = 0.

        unit_cnt = 0
        kl_unit_cnt = 0
        kl_overshoot_unit_cnt = 0

        pred_states_batch = []
        pred_kv_batch = []
        pred_rewards_batch = [[] for _ in obs]
        gt_rewards_batch = [[] for _ in obs]
        pred_img_embeds_batch = [[] for _ in obs]
        gt_img_labels_batch = [[] for _ in obs]

        pred_gsnn_rewards_batch = [[[] for _ in obs] for _ in range(5)]
        gt_gsnn_rewards_batch = [[[] for _ in obs] for _ in range(5)]
        pred_gsnn_img_embeds_batch = [[[] for _ in obs] for _ in range(5)]
        gt_gsnn_img_labels_batch = [[[] for _ in obs] for _ in range(5)]

        all_img_embeds_batch = [[] for _ in obs]
        aux_img_embeds_batch = [[] for _ in obs]
        all_rewards_batch = [[] for _ in obs]

        neg_selected_batch = [set() for _ in range(batch_size)]

        emap_nodes = [{} for _ in range(batch_size)]

        if train_ml is None:
            look_ahead_img_embeds = [[[] for _ in range(batch_size)] for _ in range(5)]
            look_ahead_rewards = [[[] for _ in range(batch_size)] for _ in range(5)]
            gt_look_ahead_indices = [[[] for _ in range(batch_size)] for _ in range(5)]

        past_key_values = None
        pred_valid_flags = []
        teleport_vps = [None for _ in obs]

        for t in range(self.args.max_action_len):
            for i, ob in enumerate(obs):
                gmap_id = gmap_id_map[i]
                if not ended[i]:
                    gmaps[gmap_id].node_step_ids[robot_id_map[i]][ob['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)
            if self.args.fix_pano_embedding:
                avg_pano_embeds = avg_pano_embeds.detach()
            pano_values = self.vln_bert('transform', {"pano_embeds": avg_pano_embeds})
            if self.args.fix_pano_value:
                pano_values = pano_values.detach()
            for i, ob in enumerate(obs):
                gmap_id = gmap_id_map[i]
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmaps[gmap_id].update_node_embed(i_vp, avg_pano_embeds[i], robot_id_map[i], rewrite=True,
                                                          value=pano_values[i])
                    # update unvisited nodes
                    for j, (i_cand_vp, i_view_id) in enumerate(
                            zip(pano_inputs['cand_vpids'][i], pano_inputs['view_ids'][i])):
                        if not gmaps[gmap_id].observed(i_cand_vp, robot_id_map[i]):
                            gmaps[gmap_id].update_node_embed(i_cand_vp, pano_embeds[i, j], robot_id_map[i],
                                                                  rewrite=False, view_id=i_view_id, redundant_view=self.args.redundant_view, observe_friendly=True)

            # world model
            if t == 0:
                if 'rssm' in self.vln_bert.vln_bert.world_model_type:
                    deter = self.vln_bert.vln_bert.lang_init(txt_embeds[:, 0, :])
                    post_outs = self.vln_bert('posterior', {"deter": deter, "obs_embed": avg_pano_embeds})
                    prev_states_fts = torch.cat([self.vln_bert.vln_bert.stoch_transform(post_outs['stoch']), deter], dim=-1)
                    state_fts = torch.cat([post_outs['stoch'], deter], dim=-1)
                elif 'tssm' in self.vln_bert.vln_bert.world_model_type:
                    deter = self.vln_bert.vln_bert.lang_init(txt_embeds[:, 0, :])
                    post_outs = self.vln_bert('posterior', {"deter": deter, "encoder_embeds": txt_embeds, "encoder_attention_mask": txt_masks, "obs_embed": avg_pano_embeds})
                    prev_states_fts = torch.cat([post_outs['stoch'], deter], dim=-1).unsqueeze(1)
                    state_fts = torch.cat([post_outs['stoch'], deter], dim=-1)
                else:
                    raise NotImplementedError
            else:
                transit_outputs = self.vln_bert("transit",
                                                {"prev_states": prev_states_fts, "obs_embed": avg_pano_embeds, "encoder_embeds": txt_embeds, "encoder_attention_mask": txt_masks,
                                                 "past_key_values": past_key_values})
                state_fts = torch.cat([transit_outputs['posterior_stoch'], transit_outputs['deter']], dim=-1)
                state_gsnn_fts = torch.cat([transit_outputs['prior_stoch'], transit_outputs['deter']], dim=-1)

                if 'rssm' in self.vln_bert.vln_bert.world_model_type:
                    prev_states_fts = torch.cat([self.vln_bert.vln_bert.stoch_transform(transit_outputs['posterior_stoch']), transit_outputs['deter']], dim=-1)
                    prev_gsnn_states_fts = torch.cat([self.vln_bert.vln_bert.stoch_transform(transit_outputs['prior_stoch']), transit_outputs['deter']], dim=-1)
                elif 'tssm' in self.vln_bert.vln_bert.world_model_type:
                    prev_states_fts = state_fts.unsqueeze(1)
                    prev_gsnn_states_fts = state_gsnn_fts.unsqueeze(1)
                else:
                    raise NotImplementedError

                past_key_values = transit_outputs['past_key_values']
                recon_outs = self.vln_bert("reconstruct", {"states": state_fts})

            # kl constraint
            if train_ml is not None:
                if t > 0:
                    valid_flags = torch.logical_and(~ended, action_flags)
                    true_valid_flags = torch.logical_and(valid_flags, not_jump_flags)
                    if len(pred_states_batch):
                        # Stack all predicted states and past key values
                        stacked_pred_states = torch.cat([states for states in pred_states_batch],
                                                        dim=0)  # [num_steps, batch_size, ...]
                        stacked_txt_embeds = torch.cat([txt_embeds for states in pred_states_batch], dim=0)
                        stacked_txt_masks = torch.cat([txt_masks for states in pred_states_batch], dim=0)
                        stacked_pred_kv = []
                        num_layers = 0
                        if 'tssm' in self.vln_bert.vln_bert.world_model_type:
                            num_layers = len(pred_kv_batch[0])

                            # For each layer
                            for layer_idx in range(num_layers):
                                # Stack keys and values separately for this layer
                                layer_k = torch.cat([step_kv[layer_idx][0] for step_kv in pred_kv_batch], dim=0)
                                layer_v = torch.cat([step_kv[layer_idx][1] for step_kv in pred_kv_batch], dim=0)
                                ca_layer_k = torch.cat([step_kv[layer_idx][2] for step_kv in pred_kv_batch], dim=0)
                                ca_layer_v = torch.cat([step_kv[layer_idx][3] for step_kv in pred_kv_batch], dim=0)
                                stacked_pred_kv.append((layer_k, layer_v, ca_layer_k, ca_layer_v))

                        # Create mask for all steps
                        num_steps = len(pred_states_batch)
                        step_valid_flags = torch.logical_and(
                            torch.stack(pred_valid_flags, dim=0),
                            true_valid_flags.unsqueeze(0)
                        )  # [num_steps, batch_size]

                        look_ahead_steps = [t - i - 1 for i in range(len(pred_states_batch))]
                        look_ahead_steps = torch.tensor(look_ahead_steps).expand(batch_size, num_steps).t().cuda()
                        step_valid_flags = torch.logical_and(step_valid_flags, look_ahead_steps < 5)

                        # Batch process all imagined states
                        imagined_outs = self.vln_bert("imagine", {
                            "cur_states": stacked_pred_states,
                            "encoder_embeds": stacked_txt_embeds, "encoder_attention_mask": stacked_txt_masks,
                            "past_key_values": stacked_pred_kv
                        })

                        # Reshape outputs back to [num_steps, batch_size, ...]
                        batch_size = true_valid_flags.shape[0]
                        imagined_dist = imagined_outs['dist']
                        imagined_stoch = imagined_outs['stoch'].view(num_steps, batch_size, -1)
                        imagined_deter = imagined_outs['deter'].view(num_steps, batch_size, -1)
                        imagined_state = torch.cat([imagined_stoch, imagined_deter], dim=-1)

                        # Reshape the output past_key_values back to tuple structure
                        imagined_kv = imagined_outs['past_key_values']
                        reshaped_kv = []
                        for layer_idx in range(num_layers):
                            layer_k, layer_v, ca_layer_k, ca_layer_v = imagined_kv[layer_idx]
                            reshaped_k = layer_k.view(num_steps, batch_size, *layer_k.shape[1:])
                            reshaped_v = layer_v.view(num_steps, batch_size, *layer_v.shape[1:])
                            reshaped_ca_k = ca_layer_k.view(num_steps, batch_size, *ca_layer_k.shape[1:])
                            reshaped_ca_v = ca_layer_v.view(num_steps, batch_size, *ca_layer_v.shape[1:])
                            reshaped_kv.append((reshaped_k, reshaped_v, reshaped_ca_k, reshaped_ca_v))

                        # Compute KL divergence for all steps at once
                        posterior_dist = zdistr(
                            transit_outputs['posterior_mu_std'].repeat(num_steps, 1),
                            self.vln_bert.vln_bert.stoch_discrete,
                            self.vln_bert.vln_bert.stoch_size
                        )
                        kl_overshoot_loss = torch.distributions.kl_divergence(posterior_dist, imagined_dist).mean(
                            dim=-1) * step_valid_flags.view(-1)
                        kl_overshoot_losses += kl_overshoot_loss.view(num_steps, batch_size).sum(dim=0)
                        kl_overshoot_unit_cnt += step_valid_flags.view(num_steps, batch_size).sum(dim=0)

                        # Batch process reconstructions
                        if self.args.stop_gradient:
                            imagined_recon_outs = self.vln_bert("reconstruct", {
                                "states": imagined_state.reshape(-1, imagined_state.shape[-1]).detach()
                            })
                        else:
                            imagined_recon_outs = self.vln_bert("reconstruct", {
                                "states": imagined_state.reshape(-1, imagined_state.shape[-1])
                            })

                        # Reshape reconstruction outputs
                        imagined_img_embeds = imagined_recon_outs['img_embed'].view(num_steps, batch_size, -1)
                        imagined_rewards = imagined_recon_outs['reward'].view(num_steps, batch_size, 1)

                        # Update prediction batches using masked operations
                        for step_idx in range(len(pred_states_batch)):
                            valid_step_mask = step_valid_flags[step_idx]

                            # Update GSNN batches
                            for i in range(batch_size):
                                look_ahead_step = t - step_idx - 1
                                assert look_ahead_step != 0
                                if valid_step_mask[i]:
                                    assert look_ahead_step < 5
                                    pred_gsnn_img_embeds_batch[look_ahead_step][i].append(imagined_img_embeds[step_idx, i])
                                    pred_gsnn_rewards_batch[look_ahead_step][i].append(imagined_rewards[step_idx, i])
                                    gt_gsnn_img_labels_batch[look_ahead_step][i].append(t)
                                    gt_gsnn_rewards_batch[look_ahead_step][i].append(rewards[i])

                                    # Handle look-ahead predictions if train_ml is None
                                    if train_ml is None:
                                        look_ahead_img_embeds[look_ahead_step][i].append(imagined_img_embeds[step_idx, i])
                                        look_ahead_rewards[look_ahead_step][i].append(imagined_rewards[step_idx, i])
                                        gt_look_ahead_indices[look_ahead_step][i].append(t)

                            # Update state and key-value batches
                            if 'rssm' in self.vln_bert.vln_bert.world_model_type:
                                pred_states_batch[step_idx] = torch.cat([self.vln_bert.vln_bert.stoch_transform(imagined_stoch[step_idx]), imagined_deter[step_idx]], dim=-1)
                            elif 'tssm' in self.vln_bert.vln_bert.world_model_type:
                                pred_states_batch[step_idx] = imagined_state[step_idx].unsqueeze(1)
                            else:
                                raise NotImplementedError
                            # Update key-value batch while maintaining tuple structure
                            step_kv = []
                            for layer_idx in range(num_layers):
                                layer_k = reshaped_kv[layer_idx][0][step_idx]
                                layer_v = reshaped_kv[layer_idx][1][step_idx]
                                ca_layer_k = reshaped_kv[layer_idx][2][step_idx]
                                ca_layer_v = reshaped_kv[layer_idx][3][step_idx]
                                step_kv.append((layer_k, layer_v, ca_layer_k, ca_layer_v))
                            pred_kv_batch[step_idx] = step_kv
                            pred_valid_flags[step_idx] = valid_step_mask

                    # gsnn
                    state_gsnn_fts = torch.cat([transit_outputs['prior_stoch'], transit_outputs['deter']], dim=-1)
                    if self.args.stop_gradient:
                        recon_gsnn_outs = self.vln_bert("reconstruct", {"states": state_gsnn_fts.detach()})
                    else:
                        recon_gsnn_outs = self.vln_bert("reconstruct", {"states": state_gsnn_fts})
                    for i in range(batch_size):
                        if true_valid_flags[i]:
                            pred_gsnn_img_embeds_batch[0][i].append(recon_gsnn_outs['img_embed'][i])
                            pred_gsnn_rewards_batch[0][i].append(recon_gsnn_outs['reward'][i])
                            gt_gsnn_img_labels_batch[0][i].append(t)
                            gt_gsnn_rewards_batch[0][i].append(rewards[i])

                            if train_ml is None:
                                look_ahead_img_embeds[0][i].append(recon_gsnn_outs['img_embed'][i])
                                look_ahead_rewards[0][i].append(recon_gsnn_outs['reward'][i])
                                gt_look_ahead_indices[0][i].append(t)

                    # kl
                    kl_losses += torch.distributions.kl_divergence(transit_outputs['posterior_dist'],
                                                                   transit_outputs['prior_dist']).mean(
                        dim=-1) * true_valid_flags
                    kl_unit_cnt += true_valid_flags

                    pred_states_batch.append(prev_gsnn_states_fts)
                    pred_kv_batch.append(past_key_values)
                    pred_valid_flags.append(valid_flags)

                    for i in range(batch_size):
                        if true_valid_flags[i]:
                            pred_rewards_batch[i].append(recon_outs['reward'][i])
                            gt_rewards_batch[i].append(rewards[i])
                            pred_img_embeds_batch[i].append(recon_outs['img_embed'][i])
                            gt_img_labels_batch[i].append(t)
                        elif mistake_flags[i]:
                            aux_img_embeds_batch[i].append(avg_pano_embeds[i])

            # extend memory
            with torch.inference_mode():
                tmp_prev_states_fts = prev_states_fts.detach()
                tmp_past_key_values = None
                fut_states_fts = []
                for _ in range(self.args.look_ahead_steps):
                    step_imagined_outs = self.vln_bert('imagine', {"cur_states": tmp_prev_states_fts, "encoder_embeds": txt_embeds, "encoder_attention_mask": txt_masks,
                                                                   "past_key_values": past_key_values if tmp_past_key_values is None else tmp_past_key_values})
                    tmp_states_fts = torch.cat([step_imagined_outs['stoch'], step_imagined_outs['deter']], dim=-1)
                    tmp_past_key_values = step_imagined_outs['past_key_values']
                    if 'tssm' in self.vln_bert.vln_bert.world_model_type:
                        tmp_prev_states_fts = tmp_states_fts.unsqueeze(1)
                    fut_states_fts.append(tmp_states_fts)

                fut_states_fts = torch.stack(fut_states_fts, dim=1)
                fut_recon_outs = self.vln_bert("reconstruct", {"states": fut_states_fts})
                # mask out after the stop action
                pred_obs_multisteps = fut_recon_outs['img_embed']
                pred_actions_multisteps = fut_recon_outs['reward'].le(0.25).squeeze(-1)
                pred_masks = torch.cumsum(pred_actions_multisteps, dim=1)
                pred_masks = torch.cumsum(pred_masks, dim=1) <= 1
                stop_flag = recon_outs['reward'].le(0.25).squeeze(-1) if t > 0 else np.zeros(batch_size)

                teacher_paths, teacher_stop_flags = self._teacher_path(obs, gmaps, gmap_id_map, robot_id_map, ended)
                for i, ob in enumerate(obs):
                    gmap = gmaps[gmap_id_map[i]]
                    if not ended[i]:
                        if not stop_flag[i]:
                            dropout_prob = self.args.env_memory_dropout if train_ml is not None else 0.0
                            beam_outs = gmap.smart_search([ob['viewpoint']], pred_obs_multisteps[i].unsqueeze(0), pred_masks[i].unsqueeze(0), self.args.min_beam_size, self.args.max_beam_size, [robot_id_map[i]], self.args.env_memory_filter, dropout_prob, self.args.env_memory_drop_replace, self.args.env_memory_gamma)
                            beam_nodes = beam_outs[0]
                            for j, step_nodes in enumerate(beam_nodes):
                                new_nodes = [node for node in step_nodes if not gmap.retrieved(node, robot_id_map[i])]
                                gmap.to_short_term(set(new_nodes), robot_id_map[i], self.args.include_neighbours)

                # historical experience
                valid_pairs = [[] for _ in range(batch_size)]
                for i, ob in enumerate(obs):
                    if not ended[i]:
                        gmap = gmaps[gmap_id_map[i]]
                        tmp_pairs = gmap.get_experience(ob['viewpoint'], ob['instr_id'])
                        for tmp_pair in tmp_pairs:
                            if not all(item[1] is None for item in tmp_pair['path'][:self.args.look_ahead_steps]):
                                valid_pairs[i].append(tmp_pair)
                exp_pairs, exp_sims, exp_traj_ids, exp_hit_cnt, exp_random_hit_cnt, exp_retrieved_path_cnt, exp_random_retrieved_path_cnt, exp_total_path_cnt = get_historical_experience_prompt(pred_obs_multisteps, pred_masks, valid_pairs, self.args.look_ahead_steps, self.args.his_memory_threshold, self.args.his_memory_gamma, self.args.max_pairs, teacher_paths, self.args.mastermind, self.args.jump_friendly)
                for i, ob in enumerate(obs):
                    if not ended[i]:
                        gmap = gmaps[gmap_id_map[i]]
                        for exp_pair, exp_traj_id in zip(exp_pairs[i], exp_traj_ids[i]):
                            path = [pair[0] for pair in exp_pair]
                            path = [ob['viewpoint']] + path
                            gmap.to_short_term(path, robot_id_map[i], self.args.include_neighbours)
                emap_nodes = self.update_emap_nodes(emap_nodes, exp_pairs, exp_sims, exp_traj_ids)

            experience_inputs, emap_teacher_masks = self._nav_emap_variable(obs, gmaps, state_fts, emap_nodes, gmap_id_map, robot_id_map, self.args.multimodal_history)

            # record robot status
            for i, ob in enumerate(obs):
                gmap = gmaps[gmap_id_map[i]]
                available_nodes = gmap.get_available_nodes(ob['viewpoint'], robot_id_map[i], 2)
                neg_selected_batch[i].update(available_nodes)

            # navigation policy
            nav_inputs, gmap_teacher_masks = self._nav_gmap_variable(obs, gmaps, gmap_id_map, robot_id_map)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'], gmap_id_map, robot_id_map
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
                'exp_bw': self.args.exp_bw
            })
            nav_inputs.update(experience_inputs)
            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)

            # update graph
            for i, ob in enumerate(obs):
                gmap_id = gmap_id_map[i]
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmaps[gmap_id].node_stop_scores[robot_id_map[i]][i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }

            nav_targets, target_names = self._teacher_action_r4r(
                obs, nav_vpids, nav_inputs['emap_vpids'], ended,
                visited_masks=gmap_teacher_masks if self.args.fusion != 'local' else None,
                imitation_learning=(self.feedback == 'teacher'), t=t, traj=traj
            )
            if train_ml is not None:
                # Supervised training
                step_loss = self.criterion(nav_logits, nav_targets)
                ml_losses += step_loss
                unit_cnt += (~ended)

            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample':  # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])
                if teleport_vps[i] is not None:
                    cpu_a_t[-1] = teleport_vps[i]

            last_vps = [ob['viewpoint'] for ob in obs]

            # Make action and get the new state
            raw_obs = self.make_equiv_action(gmaps, cpu_a_t, raw_obs, t, robot_id_map, traj)
            cpu_a_t = [None if teleport_vps[i] is not None else x for i, x in enumerate(cpu_a_t)]

            obs = []
            for i, cluster_obs in enumerate(raw_obs):
                valid_obs = []
                for j, ob in enumerate(cluster_obs):
                    if ob is not None:
                        valid_obs.append(ob)
                obs.extend(valid_obs)
            for i in range(batch_size):
                gmap_id = gmap_id_map[i]
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[gmap_id].node_stop_scores[robot_id_map[i]].items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(
                            gmaps[gmap_id].graph_pool.path(obs[i]['viewpoint'], stop_node))
                    if self.args.detailed_output:
                        for k, v in gmaps[gmap_id].node_stop_scores[robot_id_map[i]].items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }
            self._update_scanvp_cands(obs)

            # new observation and update graph
            for i, ob in enumerate(obs):
                gmap_id = gmap_id_map[i]
                if not ended[i]:
                    gmaps[gmap_id].update_graph(ob, robot_id_map[i])

            # belief and experience
            rewards = []
            stop_labels = []
            action_flags = []
            mistake_flags = []
            for i, ob in enumerate(obs):
                robot_id = robot_id_map[i]
                gmap_id = gmap_id_map[i]
                if not ended[i] and target_names[i] is not None:
                    target_path = gmaps[gmap_id].graph[robot_id].path(last_vps[i], target_names[i])
                    pred_path = gmaps[gmap_id].graph[robot_id].path(last_vps[i], ob['viewpoint'])
                if ob['viewpoint'] == last_vps[i]:
                    stop_labels.append(True)
                else:
                    stop_labels.append(False)
                rewards.append(self.env.shortest_distances[ob['scan']][ob['viewpoint']][ob['gt_path'][-1]])
                if target_names[i] is not None and (target_names[i] in pred_path or ob['viewpoint'] in target_path):
                    action_flags.append(True)
                else:
                    action_flags.append(False)
                if target_names[i] is not None and (target_names[i] not in pred_path and ob['viewpoint'] not in target_path):
                    mistake_flags.append(True)
                else:
                    mistake_flags.append(False)
            action_flags = torch.BoolTensor(action_flags).to(txt_embeds.device)
            if self.args.jump_friendly:
                not_jump_flags = torch.BoolTensor([not stop_labels[i] for
                     i, (ob, last_vp) in
                     enumerate(zip(obs, last_vps))]).to(txt_embeds.device)
            else:
                not_jump_flags = torch.BoolTensor(
                    [ob['viewpoint'] in self.scanvp_cands[f"{ob['scan']}_{last_vp}"] if not stop_labels[i] else True for
                     i, (ob, last_vp) in
                     enumerate(zip(obs, last_vps))]).to(txt_embeds.device)

            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmap_id = gmap_id_map[i]
                    robot_id = robot_id_map[i]
                    aux_path = gmaps[gmap_id].path(last_vps[i], ob['viewpoint'], robot_id)[:-1]
                    if t > 0:
                        gmaps[gmap_id].add_experience(last_vps[i], ob['instr_id'], [state_fts[i].detach(), pred_obs_multisteps[i].detach(), recon_outs['img_embed'][i].detach()], pred_masks[i], aux_path)
                    else:
                        gmaps[gmap_id].add_experience(last_vps[i], ob['instr_id'], None, None, aux_path)

            for i in range(batch_size):
                if not ended[i]:
                    all_img_embeds_batch[i].append(avg_pano_embeds[i])
                    all_rewards_batch[i].append(rewards[i])

            ended[:] = torch.logical_or(ended, torch.BoolTensor([x is None for x in cpu_a_t]).to(txt_embeds.device))

            # Early exit if all ended
            if ended.all():
                break

        full_all_img_embeds, full_pred_img_embeds, full_gt_img_labels, full_pred_rewards, full_gt_rewards = [], [], [], [], []
        full_pred_gsnn_img_embeds = []
        full_gt_gsnn_img_labels = []
        full_pred_gsnn_rewards = []
        full_gt_gsnn_rewards = []

        cum_length = 0
        for i in range(batch_size):
            full_all_img_embeds.extend(all_img_embeds_batch[i])
            full_pred_img_embeds.extend(pred_img_embeds_batch[i])
            full_gt_img_labels.extend([index+cum_length for index in gt_img_labels_batch[i]])
            full_pred_rewards.extend(pred_rewards_batch[i])
            full_gt_rewards.extend(gt_rewards_batch[i])
            for j in range(5):
                full_pred_gsnn_img_embeds.extend(pred_gsnn_img_embeds_batch[j][i])
                full_gt_gsnn_img_labels.extend([index+cum_length for index in gt_gsnn_img_labels_batch[j][i]])
                full_pred_gsnn_rewards.extend(pred_gsnn_rewards_batch[j][i])
                full_gt_gsnn_rewards.extend(gt_gsnn_rewards_batch[j][i])
            cum_length += len(all_img_embeds_batch[i])
        full_all_img_embeds = torch.stack(full_all_img_embeds, dim=0)
        full_pred_img_embeds = torch.stack(full_pred_img_embeds, dim=0) if len(full_pred_img_embeds) > 0 else None
        full_gt_img_labels = torch.LongTensor(full_gt_img_labels).to(txt_embeds.device) if len(full_gt_img_labels) > 0 else None
        full_pred_rewards = torch.stack(full_pred_rewards, dim=0) if len(full_pred_rewards) > 0 else None
        full_gt_rewards = torch.FloatTensor(full_gt_rewards).to(txt_embeds.device) if len(full_gt_rewards) > 0 else None
        full_pred_gsnn_img_embeds = torch.stack(full_pred_gsnn_img_embeds, dim=0) if len(full_pred_gsnn_img_embeds) > 0 else None
        full_gt_gsnn_img_labels = torch.LongTensor(full_gt_gsnn_img_labels).to(txt_embeds.device) if len(full_gt_gsnn_img_labels) > 0 else None
        full_pred_gsnn_rewards = torch.stack(full_pred_gsnn_rewards, dim=0) if len(full_pred_gsnn_rewards) > 0 else None
        full_gt_gsnn_rewards = torch.FloatTensor(full_gt_gsnn_rewards).to(txt_embeds.device) if len(full_gt_gsnn_rewards) > 0 else None

        aux_all_img_embeds = []
        for i in range(batch_size):
            aux_all_img_embeds.extend(aux_img_embeds_batch[i])
        if len(aux_all_img_embeds) > 0:
            aux_all_img_embeds = torch.stack(aux_all_img_embeds, dim=0)

        all_neg_img_embeds_batch = [[] for _ in obs]
        all_neg_img_embeds = []
        all_img_embed_lengths = []
        for i in range(batch_size):
            all_img_embed_lengths.append(len(all_img_embeds_batch[i]))
            neg_set = neg_selected_batch[i]
            for node in traj[i]['gt_path']:
                neg_set.discard(node)
            for part in traj[i]['path']:
                for node in part:
                    neg_set.discard(node)
            all_neg_img_embeds_batch[i].extend([gmaps[gmap_id_map[i]].node_embeds_pool[vp][0] for vp in neg_set])
            all_neg_img_embeds.extend([gmaps[gmap_id_map[i]].node_embeds_pool[vp][0] for vp in neg_set])
            assert sum([gmaps[gmap_id_map[i]].node_embeds_pool[vp][1] for vp in neg_set]) == 0
        if len(all_neg_img_embeds) > 0:
            all_neg_img_embeds = torch.stack(all_neg_img_embeds, dim=0)

        # loss
        if train_ml is not None:
            if len(aux_all_img_embeds) > 0:
                full_all_img_embeds = torch.cat([full_all_img_embeds, aux_all_img_embeds], dim=0)
            if len(all_neg_img_embeds) > 0:
                full_all_img_embeds = torch.cat([full_all_img_embeds, all_neg_img_embeds], dim=0)
            full_all_img_embeds = self.vln_bert("transform", {"pano_embeds": full_all_img_embeds})
            if self.args.fix_pano_value:
                full_all_img_embeds = full_all_img_embeds.detach()

            if full_pred_img_embeds is not None:
                if self.args.info_nce_reduction == "sum":
                    recon_reward_losses = F.mse_loss(full_pred_rewards.view(-1), full_gt_rewards, reduction="sum") / batch_size
                    recon_vis_losses = info_nce_loss(full_pred_img_embeds, full_all_img_embeds, full_gt_img_labels, temperature=self.args.info_nce_temperature, reduction="sum") / batch_size
                elif self.args.info_nce_reduction == "mean":
                    recon_reward_losses = F.mse_loss(full_pred_rewards.view(-1), full_gt_rewards, reduction="mean")
                    recon_vis_losses = info_nce_loss(full_pred_img_embeds, full_all_img_embeds, full_gt_img_labels, temperature=self.args.info_nce_temperature, reduction="mean")
                else:
                    raise Exception()

            if full_pred_gsnn_img_embeds is not None:
                if self.args.info_nce_reduction == "sum":
                    recon_reward_gsnn_losses = F.mse_loss(full_pred_gsnn_rewards.view(-1), full_gt_gsnn_rewards, reduction="sum") / (batch_size*5)
                    recon_vis_gsnn_losses = info_nce_loss(full_pred_gsnn_img_embeds, full_all_img_embeds, full_gt_gsnn_img_labels, temperature=self.args.info_nce_temperature, reduction="sum") / (batch_size*5)
                elif self.args.info_nce_reduction == "mean":
                    recon_reward_gsnn_losses = F.mse_loss(full_pred_gsnn_rewards.view(-1), full_gt_gsnn_rewards, reduction="mean")
                    recon_vis_gsnn_losses = info_nce_loss(full_pred_gsnn_img_embeds, full_all_img_embeds, full_gt_gsnn_img_labels, temperature=self.args.info_nce_temperature, reduction="mean")

            if isinstance(kl_unit_cnt, torch.Tensor) and kl_unit_cnt.sum() > 0:
                kl_losses = kl_losses.sum() / kl_unit_cnt.sum()
            else:
                kl_losses = 0.

            if isinstance(kl_overshoot_unit_cnt, torch.Tensor) and kl_overshoot_unit_cnt.sum() > 0:
                kl_overshoot_losses = kl_overshoot_losses.sum() / kl_overshoot_unit_cnt.sum()
            else:
                kl_overshoot_losses = 0.

            if self.args.loss_unit == 'step':
                ml_losses = ml_losses.sum() / unit_cnt.sum()
            elif self.args.loss_unit == 'batch':
                # ml_losses = (ml_losses / (unit_cnt + 1e-8)).mean(-1)
                ml_losses = ml_losses.mean(-1)
            else:
                raise Exception()

        # imagine metrics
        if train_ml is None:
            for i in range(batch_size):
                all_img_embeds_batch[i] = torch.stack(all_img_embeds_batch[i], dim=0)
                if len(aux_img_embeds_batch[i]) > 0:
                    aux_img_embeds_batch[i] = torch.stack(aux_img_embeds_batch[i], dim=0)
                    all_img_embeds_batch[i] = torch.cat([all_img_embeds_batch[i], aux_img_embeds_batch[i]], dim=0)
                if len(all_neg_img_embeds_batch[i]) > 0:
                    all_neg_img_embeds_batch[i] = torch.stack(all_neg_img_embeds_batch[i], dim=0)
                    all_img_embeds_batch[i] = torch.cat([all_img_embeds_batch[i], all_neg_img_embeds_batch[i]], dim=0)
                all_img_embeds_batch[i] = self.vln_bert("transform", {"pano_embeds": all_img_embeds_batch[i]})
                if self.args.fix_pano_value:
                    all_img_embeds_batch[i] = all_img_embeds_batch[i].detach()
            for i in range(5):
                for j in range(batch_size):
                    if len(look_ahead_img_embeds[i][j]) > 0:
                        look_ahead_rewards_tmp = torch.stack(look_ahead_rewards[i][j], dim=0)
                        reward_preds_tmp = look_ahead_rewards_tmp.view(-1)
                        traj[j]['imagined_actions'][i] = reward_preds_tmp.tolist()
                        rewards = [all_rewards_batch[j][idx] for idx in gt_look_ahead_indices[i][j]]
                        traj[j]['action_labels'][i] = rewards
                        img_logits = torch.matmul(torch.stack(look_ahead_img_embeds[i][j]), all_img_embeds_batch[j].t())
                        traj[j]['imagined_obs'][i] = torch.argmax(img_logits, dim=-1).tolist()
                        traj[j]['obs_labels'][i] = gt_look_ahead_indices[i][j]
                        traj[j]['obs_soft_labels'][i] = all_img_embed_lengths[j]

        if train_ml is not None:
            loss = ml_losses * train_ml + kl_losses * self.args.kl_weight + kl_overshoot_losses * self.args.kl_overshoot_weight + 0.1 * recon_reward_losses * self.args.recon_act_weight + recon_vis_losses * self.args.recon_vis_weight + 0.1 * recon_reward_gsnn_losses * self.args.recon_act_gsnn_weight + recon_vis_gsnn_losses * self.args.recon_vis_gsnn_weight
            try:
                self.loss += loss
            except Exception:
                print(loss, ml_losses, kl_losses, kl_overshoot_losses, recon_reward_losses, recon_vis_losses,
                      recon_reward_gsnn_losses, recon_vis_gsnn_losses)
            self.logs['IL_loss'].append(ml_losses.item())
            self.logs['KL_loss'].append(kl_losses if isinstance(kl_losses, float) else kl_losses.item())
            self.logs['KL_overshoot_loss'].append(
                kl_overshoot_losses if isinstance(kl_overshoot_losses, float) else kl_overshoot_losses.item())
            self.logs['REACT_loss'].append(
                recon_reward_gsnn_losses if isinstance(recon_reward_gsnn_losses, float) else recon_reward_gsnn_losses.item())
            self.logs['REVIS_loss'].append(
                recon_vis_losses if isinstance(recon_vis_losses, float) else recon_vis_losses.item())

        return traj
