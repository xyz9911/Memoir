'''
Instruction and trajectory (view and object features) dataset
'''
import copy
import os
import json
import random

import jsonlines
import numpy as np
import h5py
import math
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

from .common import load_nav_graphs, pad_tensors, pad_tensors_wgrad, gen_seq_masks_wgrad
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax

from utils.data import new_simulator

MAX_DIST = 30  # normalize
MAX_STEP = 10  # normalize
TRAIN_MAX_STEP = 20


class ReverieTextPathData(object):
    def __init__(
            self, anno_file, aug_anno_file, img_ft_file, obj_ft_file, sent_embed_file, scanvp_cands_file,
            connectivity_dir,
            image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
            obj_feat_size=None, obj_prob_size=None, max_objects=20,
            max_txt_len=500, in_memory=True, act_visited_node=False
    ):
        self.img_ft_file = img_ft_file
        self.obj_ft_file = obj_ft_file
        self.sent_embed_file = sent_embed_file

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}
            self._embed_store = defaultdict(dict)

        # {scan_vp: {vp: [viewidx, rel_angle_dist, rel_heading, rel_elevation]}}
        self.scanvp_cands = json.load(open(scanvp_cands_file))
        self.connectivity_dir = connectivity_dir

        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in
                                    self.all_point_rel_angles]

        self.data = []
        with open(anno_file, 'r') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                for j, instr in enumerate(item['instructions']):
                    # if j >= len(item['sub_instructions']):
                    #     continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instruction'] = instr
                    new_item['instr_encoding'] = item['instr_encodings'][j]
                    # new_item['chunk_view'] = item['chunk_views'][j]
                    # if len(new_item['chunk_view']) == 1:
                    #     continue
                    del new_item['instructions']
                    del new_item['instr_encodings']
                    del new_item['sub_instructions']
                    del new_item['headings']
                    del new_item['chunk_views']
                    self.data.append(new_item)

        if aug_anno_file:
            with jsonlines.open(aug_anno_file, 'r') as f:
                for item in f:
                    self.data.append(item)
        random.shuffle(self.data)

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
            sim = new_simulator(self.connectivity_dir)
            sim.newEpisode([scan], [path[0]], [heading], [elevation])
            states = sim.getState()
            viewidx = states[0].viewIndex
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s' % (scan, prev_vp)][cur_vp][0]
        heading = (viewidx % 12) * math.radians(30)
        elevation = (viewidx // 12 - 1) * math.radians(30)
        return heading, elevation, viewidx

    def get_batch_cur_angles(self, scans, paths, start_headings):
        batch_headings, batch_elevations, batch_viewidxs = [], [], []
        for scan, path, start_heading in zip(scans, paths, start_headings):
            heading, elevation, viewidx = self.get_cur_angle(scan, path, start_heading)
            batch_headings.append(heading)
            batch_elevations.append(elevation)
            batch_viewidxs.append(viewidx)
        return batch_headings, batch_elevations, batch_viewidxs

    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s' % (scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        gmap_node_types = [0] + [1] * len(visited_vpids) + [3] * len(unvisited_vpids)
        if self.act_visited_node:
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # shape=(num_gmap_vpids, 7)
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)

        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i + 1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]

        return gmap_vpids, gmap_step_ids, gmap_node_types, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists

    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'],
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                     (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)

        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len + 1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts

        return vp_pos_fts


class R2RTextPathData(ReverieTextPathData):
    def __init__(
            self, anno_file, aug_anno_file, img_ft_file, sent_embed_file, scanvp_cands_file, connectivity_dir,
            image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
            max_txt_len=500, in_memory=True, act_visited_node=False
    ):
        super().__init__(
            anno_file, aug_anno_file, img_ft_file, None, sent_embed_file, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=0, obj_prob_size=0,
            max_objects=0, max_txt_len=max_txt_len, in_memory=in_memory,
            act_visited_node=act_visited_node
        )

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store[key] = view_fts
        return view_fts

    def get_act_labels(self, end_vp, end_idx, path, gmap_vpids, traj_cand_vpids):
        if end_vp == path[-1]:  # stop
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = path[end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local: 
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1  # [stop] is 0
                    break
        return global_act_label, local_act_label

    def get_input_sr(
            self, idx
    ):
        item = self.data[idx]
        instr_id = item['instr_id']
        instr_encoding = item['instr_encoding']
        scan = item['scan']
        gt_path = item['path']
        heading = item['heading']
        # sub_instr_embed = []
        # chunk_view = []
        # for k, chunk in enumerate(item['chunk_view']):
        #     if chunk[0] != chunk[1]:
        #         chunk_view.append(chunk)
        #         sub_instr_embed.append(self.get_sent_embed(instr_id, k))
        #
        # chunk = random.choice(chunk_view)

        chunk_ix = random.randint(1, len(gt_path) - 1)
        path = gt_path[:chunk_ix + 1]

        # sample_start = random.randint(0, chunk[0]-2)
        # sample_end = random.randint(chunk[0], len(path)-1)
        # neg_path = path[:sample_start+1]
        # for j in range(sample_start, sample_end):
        #     vp = neg_path[-1]
        #     nav_cands = self.scanvp_cands['%s_%s' % (scan, vp)].keys()
        #     nav_cands = [v for v in nav_cands if v not in path]
        #     tgt = random.choice(nav_cands)
        #     neg_path.append(tgt)

        out = {
            'scan': scan,
            'instr_encoding': instr_encoding,
            'heading': heading,
            'path': path,
            'gt_path': gt_path,
        }
        return out

    def get_input_sap(
            self, idx, end_vp_type, return_img_probs=False, return_act_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item['heading']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos':
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1]
                end_idx = np.random.randint(len(end_vps))
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)

        gt_path = gt_path[:end_idx + 1]
        cur_heading, cur_elevation, cur_view_idx = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]

        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_node_types, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
                                         traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],

            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_node_types': gmap_node_types,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            'vp_angles': last_vp_angles,
        }

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item['path'], gmap_vpids, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)

        return outs

    def get_input(
            self, idx, end_vp_type, return_img_probs=False, return_act_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item['heading']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos':
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1]
                end_idx = np.random.randint(len(end_vps))
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)

        gt_path = gt_path[:end_idx + 1]
        cur_heading, cur_elevation, cur_view_idx = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]

        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_node_types, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
                                         traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],

            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_node_types': gmap_node_types,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            'vp_angles': last_vp_angles,
        }

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item['path'], gmap_vpids, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)

        return outs

    def get_batch_pano_fts(self, scans, viewpoints, view_ids):
        batch_view_img_fts, batch_loc_fts, batch_nav_types, batch_cand_vpids = [], [], [], []
        batch_view_lens = []

        for i, (scan, vp) in enumerate(zip(scans, viewpoints)):
            view_img_fts, loc_fts, cand_vpids, view_angles = [], [], [], []
            view_fts = self.get_scanvp_feature(scan, vp)
            # cand views
            nav_cands = self.scanvp_cands['%s_%s' % (scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_angle = self.all_point_rel_angles[view_ids[i]][v[0]]
                # view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                view_angles.append(view_angle)
                cand_vpids.append(k)

            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[view_ids[i]][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            nav_types = [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs))

            batch_view_img_fts.append(torch.from_numpy(view_img_fts[:, :self.image_feat_size]))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return batch_view_img_fts, batch_loc_fts, batch_nav_types, batch_view_lens, batch_cand_vpids

    def get_negative_pano_fts(self, scans, viewpoints, gt_paths, selects):
        batch_view_img_fts, batch_loc_fts, batch_nav_types, batch_cand_vpids = [], [], [], []
        batch_view_lens = []
        batch_sources = []

        for i, (scan, vp, gt_path) in enumerate(zip(scans, viewpoints, gt_paths)):
            neighbours = self.graphs[scan].neighbors(vp)
            neighbours = [n for n in neighbours if n not in gt_path and n not in selects]
            for neighbour_vp in neighbours:
                view_img_fts, loc_fts, cand_vpids, view_angles = [], [], [], []
                view_fts = self.get_scanvp_feature(scan, neighbour_vp)
                # cand views
                nav_cands = self.scanvp_cands['%s_%s' % (scan, neighbour_vp)]
                used_viewidxs = set()
                for k, v in nav_cands.items():
                    used_viewidxs.add(v[0])
                    view_img_fts.append(view_fts[v[0]])
                    view_angle = self.all_point_rel_angles[12][v[0]]
                    # view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                    view_angles.append(view_angle)
                    cand_vpids.append(k)

                view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
                view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
                # combine cand views and noncand views
                view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
                view_angles = np.stack(view_angles, 0)
                view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
                view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
                view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
                nav_types = [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs))

                batch_view_img_fts.append(torch.from_numpy(view_img_fts[:, :self.image_feat_size]))
                batch_loc_fts.append(torch.from_numpy(view_loc_fts))
                batch_nav_types.append(torch.LongTensor(nav_types))
                batch_cand_vpids.append(cand_vpids)
                batch_view_lens.append(len(view_img_fts))

                selects.add(neighbour_vp)
                batch_sources.append(i)

        # pad features to max_len
        if len(batch_view_img_fts) == 0:
            return None, None, None, None, None, batch_sources, selects

        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return batch_view_img_fts, batch_loc_fts, batch_nav_types, batch_view_lens, batch_cand_vpids, batch_sources, selects

    def get_batch_gmap_variables(self, cur_vps, cur_headings, cur_elevations, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(cur_vps)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_gmap_node_types = []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if self.act_visited_node:
                    if k == cur_vps[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            gmap_vpids = [None] + visited_vpids + unvisited_vpids
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_node_types = [0] + [1] * len(visited_vpids) + [3] * len(unvisited_vpids)

            if len(gmap_vpids) != 1:
                gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
                gmap_img_embeds = torch.stack(
                    [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
                )  # cuda
            else:
                gmap_img_embeds = torch.stack(
                    [torch.zeros(self.image_feat_size)], 0
                ).cuda()

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vps[i], gmap_vpids, cur_headings[i], cur_elevations[i],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_node_types.append(torch.LongTensor(gmap_node_types))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks_wgrad(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_node_types = pad_sequence(batch_gmap_node_types, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return batch_gmap_vpids, batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_node_types, batch_gmap_pos_fts, batch_gmap_visited_masks, gmap_pair_dists, batch_gmap_masks, batch_no_vp_left

    def get_batch_vp_variables(self, cur_vps, cur_headings, cur_elevations, gmaps, pano_embeds, cand_vpids, view_lens,
                               nav_types):
        batch_size = len(cur_vps)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                cur_vps[i], cand_vpids[i],
                cur_headings[i], cur_elevations[i]
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                cur_vps[i], [gmap.start_vp],
                cur_headings[i], cur_elevations[i]
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()
        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return vp_img_embeds, batch_vp_pos_fts, gen_seq_masks_wgrad(view_lens + 1), vp_nav_masks, [[None] + x for x in cand_vpids]

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], []

        for vp in path:
            view_fts = self.get_scanvp_feature(scan, vp)
            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s' % (scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_loc_fts.append(np.concatenate([view_ang_fts, view_box_fts], 1))
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)))
            traj_cand_vpids.append(cand_vpids)

            last_vp_angles = view_angles

        return traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles
