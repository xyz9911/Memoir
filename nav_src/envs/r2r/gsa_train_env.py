''' Batched REVERIE navigation environment '''

import h5py
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict
import copy

import MatterSim

from utils.data import load_nav_graphs, new_simulator
from utils.data import angle_feature, get_all_point_angle_feature

from nav_src.envs.eval_utils import cal_dtw, cal_cls

ERROR_MARGIN = 3.0


class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, connectivity_dir, scan_data_dir=None, feat_db=None, num_robots=1, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60

        self.sims = []
        for i in range(batch_size):
            sims_scan = []
            for j in range(num_robots):
                sim = MatterSim.Simulator()
                if scan_data_dir:
                    sim.setDatasetPath(scan_data_dir)
                sim.setNavGraphPath(connectivity_dir)
                sim.setRenderingEnabled(False)
                sim.setDiscretizedViewingAngles(True)  # Set increment/decrement to 30 degree. (otherwise by radians)
                sim.setCameraResolution(self.image_w, self.image_h)
                sim.setCameraVFOV(math.radians(self.vfov))
                sim.setBatchSize(1)
                sim.initialize()
                sim.newRandomEpisode(['2t7WUuJeko7'])
                sims_scan.append(sim)
            self.sims.append(sims_scan)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanIds_scan, viewpointIds_scan, headings_scan) in enumerate(zip(scanIds, viewpointIds, headings)):
            for j, (scanId, viewpointId, heading) in enumerate(zip(scanIds_scan, viewpointIds_scan, headings_scan)):
                if scanId is not None and viewpointId is not None and heading is not None:
                    self.sims[i][j].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim_scan in enumerate(self.sims):
            states = []
            for j, sim in enumerate(sim_scan):
                state = sim.getState()[0]
                feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId)
                states.append((feature, state))
            feature_states.append(states)
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, actions_scan in enumerate(actions):
            for j, (index, heading, elevation) in enumerate(actions_scan):
                self.sims[i][j].makeAction([index], [heading], [elevation])


class GSATrainR2RNavBatch(object):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
            self, view_db, instr_data, connectivity_dir, num_robots,
            batch_size=64, angle_feat_size=4, seed=0, name=None
    ):
        self.env = EnvBatch(connectivity_dir, feat_db=view_db, num_robots=num_robots, batch_size=batch_size)
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.name = name

        self.env_traj_num = defaultdict(int)
        for d in self.data:
            self.env_traj_num[d['scan']] += 1

        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.gt_trajs = self._get_gt_trajs(self.data)  # for evaluation

        self.data = self.scale_up(self.data)
        self.tour_ended = np.array([False] * self.batch_size)

        self.num_robots = num_robots
        self.ix = 0
        self._load_nav_graphs()

        self.sim = new_simulator(self.connectivity_dir)
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)

        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def scale_up(self, data):
        full_data = [[d] for d in data]
        return full_data

    def _get_gt_trajs(self, data):
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
            for x in data if len(x['path']) > 1
        }
        return gt_trajs

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        hash_count = 0
        self.hash_node_map = {}
        self.hash_distance_map = {}
        for scan, scan_shortest in self.shortest_distances.items():
            for s_node, node_shortest in scan_shortest.items():
                if scan + '_' + s_node not in self.hash_node_map:
                    self.hash_node_map[scan + '_' + s_node] = hash_count
                    s_hash = hash_count
                    hash_count += 1
                else:
                    s_hash = self.hash_node_map[scan + '_' + s_node]
                for t_node, distance in node_shortest.items():
                    if scan + '_' + t_node not in self.hash_node_map:
                        self.hash_node_map[scan + '_' + t_node] = hash_count
                        t_hash = hash_count
                        hash_count += 1
                    else:
                        t_hash = self.hash_node_map[scan + '_' + t_node]
                    self.hash_distance_map[s_hash * 100000 + t_hash] = distance

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.data[self.ix:self.ix + batch_size]
        if len(batch) < batch_size:
            if 'train' in self.name or 'aug' in self.name:
                self.reset_epoch(shuffle=True)
            else:
                self.reset_epoch(shuffle=False)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix = self.ix + batch_size

        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation
                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)
                    # Heading and elevation for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId,  # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'position': (loc.x, loc.y, loc.z),
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                 ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId',
                  'pointId', 'idx', 'position']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                visual_feat = feature[ix]
                c_new['heading'] = c_new['normalized_heading'] - base_heading
                c_new['elevation'] = c_new['normalized_elevation'] - base_elevation
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self, t=None, shortest_teacher=False):
        obs = []
        states = self.env.getStates()
        for i, cluster in enumerate(self.batch):
            cluster_states = states[i]
            cluster_obs = []
            for j, item in enumerate(cluster):
                if item is None:
                    cluster_obs.append(None)
                    continue
                feature, state = cluster_states[j]
                base_view_id = state.viewIndex

                # Full features
                candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
                # [visual_feature, angle_feature] for views
                feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

                base_heading = (base_view_id % 12) * math.radians(30)
                base_elevation = (base_view_id // 12 - 1) * math.radians(30)

                ob = {
                    'instr_id': item['instr_id'],
                    'scan': state.scanId,
                    'viewpoint': state.location.viewpointId,
                    'viewIndex': state.viewIndex,
                    'position': (state.location.x, state.location.y, state.location.z),
                    # 'heading': state.heading,
                    # 'elevation': state.elevation,
                    'heading': base_heading,
                    'elevation': base_elevation,
                    'feature': feature,
                    'candidate': candidate,
                    'navigableLocations': state.navigableLocations,
                    'instruction': item['instruction'],
                    'teacher': self._teacher_path_action(state, item['path'], t=t, shortest_teacher=shortest_teacher),
                    'instr_encoding': item['instr_encoding'],
                    'gt_path': item['path'],
                    'path_id': item['path_id'],
                    # 'sub_instr_embed': sub_instr_embed,
                    # 'chunk_view': chunk_view,
                }
                # RL reward. The negative distance between the state and the final state
                # There are multiple gt end viewpoints on REVERIE.
                if ob['instr_id'] in self.gt_trajs:
                    ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
                else:
                    ob['distance'] = 0
                cluster_obs.append(ob)
            obs.append(cluster_obs)
        return obs

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId  # Just stop here
        path = self.shortest_paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def _teacher_path_action(self, state, path, t=None, shortest_teacher=False):
        if shortest_teacher:
            return self._shortest_path_action(state, path[-1])

        teacher_vp = None
        if t is not None:
            teacher_vp = path[t + 1] if t < len(path) - 1 else state.location.viewpointId
        else:
            if state.location.viewpointId in path:
                cur_idx = path.index(state.location.viewpointId)
                if cur_idx == len(path) - 1:  # STOP
                    teacher_vp = state.location.viewpointId
                else:
                    teacher_vp = path[cur_idx + 1]
        return teacher_vp

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)
        viewpointIds = []
        headings = []
        scanIds = []
        for i, cluster in enumerate(self.batch):
            scan_scanIds = [item['scan'] if item is not None else None for item in cluster]
            scan_viewpointIds = [item['path'][0] if item is not None else None for item in cluster]
            scan_headings = [item['heading'] if item is not None else None for item in cluster]
            scanIds.append(scan_scanIds)
            viewpointIds.append(scan_viewpointIds)
            headings.append(scan_headings)
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs(t=0), None

    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, scan, pred_path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        metrics['fut_obs_accs'] = []
        metrics['fut_stop_accs'] = []

        n_stop_corrects, n_stop_totals = [0 for _ in range(5)], [0 for _ in range(5)]
        n_img_corrects, n_img_totals = [0 for _ in range(5)], [0 for _ in range(5)]
        n_img_soft_corrects, n_img_soft_totals = [0 for _ in range(5)], [0 for _ in range(5)]

        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)

            metrics['instr_id'].append(instr_id)
            metrics['scan'].append(item['scan'])
            metrics['gt_path'].append(item['gt_path'])
            metrics['agent_path'].append(item['trajectory'])
            metrics['gt_length'].append(item['gt_length'])
            for i, (pred_obs, obs_label) in enumerate(zip(item['imagined_obs'], item['obs_labels'])):
                if len(pred_obs):
                    cnt = 0
                    for pred, label in zip(pred_obs, obs_label):
                        if pred == label:
                            cnt += 1
                    n_img_corrects[i] += cnt
                    n_img_totals[i] += len(obs_label)
            for i, (pred_obs, obs_soft_label) in enumerate(zip(item['imagined_obs'], item['obs_soft_labels'])):
                if len(pred_obs):
                    cnt = 0
                    for pred in pred_obs:
                        if pred < obs_soft_label:
                            cnt += 1
                    n_img_soft_corrects[i] += cnt
                    n_img_soft_totals[i] += len(pred_obs)
            for i, (pred_action, action_label) in enumerate(zip(item['imagined_actions'], item['action_labels'])):
                if len(pred_action):
                    cnt = 0
                    for pred, label in zip(pred_action, action_label):
                        if (pred - label) < 0.5:
                            cnt += 1
                    n_stop_corrects[i] += cnt
                    n_stop_totals[i] += len(action_label)

        for i in range(5):
            if n_img_totals[i] > 0:
                metrics['fut_obs_accs'].append(n_img_corrects[i] / n_img_totals[i])
            else:
                metrics['fut_obs_accs'].append(0)
            if n_stop_totals[i] > 0:
                metrics['fut_stop_accs'].append(n_stop_corrects[i] / n_stop_totals[i])
            else:
                metrics['fut_stop_accs'].append(0)
            if n_img_soft_totals[i] > 0:
                metrics['fut_soft_obs_accs'].append(n_img_soft_corrects[i] / n_img_soft_totals[i])
            else:
                metrics['fut_soft_obs_accs'].append(0)

        avg_metrics = {
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'fut_obs_acc': [acc * 100 for acc in metrics['fut_obs_accs']],
            'fut_soft_obs_acc': [acc * 100 for acc in metrics['fut_soft_obs_accs']],
            'fut_stop_acc': [acc * 100 for acc in metrics['fut_stop_accs']],
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
        }
        metrics.pop('fut_obs_accs')
        metrics.pop('fut_soft_obs_accs')
        metrics.pop('fut_stop_accs')

        return avg_metrics, metrics
