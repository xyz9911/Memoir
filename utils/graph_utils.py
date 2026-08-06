import copy
import random
from collections import defaultdict

import math
import numpy as np
import torch

import torch.nn.functional as F

MAX_DIST = 30
MAX_STEP = 10


def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return dist


def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx ** 2 + dy ** 2), 1e-8)
    xyz_dist = max(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx / xy_dist)  # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist


def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts


class FloydGraph:
    """Floyd-Warshall with vectorized numpy operations"""

    def __init__(self, max_nodes=1000):
        self._node_to_idx = {}
        self._idx_to_node = {}
        self._node_count = 0
        self._max_nodes = max_nodes

        # Pre-allocate numpy arrays for better performance
        self._dist_matrix = np.full((max_nodes, max_nodes), 95959590, dtype=np.float32)
        np.fill_diagonal(self._dist_matrix, 0)

        self._visited = set()
        self._retrieved = set()
        self._observed = set()

        # Original dict interface for compatibility
        self._dis = {}
        self._point = {}

        self.neighbours = defaultdict(set)

    def _get_or_create_idx(self, node):
        """Map node to index, creating new mapping if needed"""
        if node not in self._node_to_idx:
            if self._node_count >= self._max_nodes:
                self._expand_matrices()
            self._node_to_idx[node] = self._node_count
            self._idx_to_node[self._node_count] = node
            self._node_count += 1
        return self._node_to_idx[node]

    def _expand_matrices(self):
        """Expand matrices when we run out of space"""
        new_size = self._max_nodes * 2
        new_dist = np.full((new_size, new_size), 95959590, dtype=np.float32)

        new_dist[:self._max_nodes, :self._max_nodes] = self._dist_matrix
        np.fill_diagonal(new_dist, 0)

        self._dist_matrix = new_dist
        self._max_nodes = new_size

    def add_edge(self, x, y, dist):
        """Add edge to graph"""
        idx_x = self._get_or_create_idx(x)
        idx_y = self._get_or_create_idx(y)

        if dist < self._dist_matrix[idx_x, idx_y]:
            self._dist_matrix[idx_x, idx_y] = dist
            self._dist_matrix[idx_y, idx_x] = dist

            # Update dict interface
            if x not in self._dis:
                self._dis[x] = {}
                self._point[x] = {}
            if y not in self._dis:
                self._dis[y] = {}
                self._point[y] = {}

            self._dis[x][y] = dist
            self._dis[y][x] = dist
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k, visit=True):
        """Vectorized Floyd-Warshall update for single node k"""
        if k not in self._node_to_idx:
            k_idx = self._get_or_create_idx(k)
        else:
            k_idx = self._node_to_idx[k]

        n = self._node_count

        # Extract relevant slices (only up to actual node count)
        dist_slice = self._dist_matrix[:n, :n]

        # Vectorized update using broadcasting
        # dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        # Get distances from all nodes to k and from k to all nodes
        dist_to_k = dist_slice[:, k_idx:k_idx+1]  # Shape (n, 1)
        dist_from_k = dist_slice[k_idx:k_idx+1, :]  # Shape (1, n)

        # Compute all possible paths through k
        dist_via_k = dist_to_k + dist_from_k  # Broadcasting: (n, 1) + (1, n) = (n, n)

        # Find where going through k is better
        improved = dist_via_k < dist_slice

        # Update distances
        self._dist_matrix[:n, :n] = np.where(improved, dist_via_k, dist_slice)

        # Update predecessors where improvements were made
        if np.any(improved):
            # Where we improved, the predecessor of j when coming from i through k is k
            improved_indices = np.where(improved)
            for i, j in zip(improved_indices[0], improved_indices[1]):
                if i != j and i != k_idx and j != k_idx:

                    # Update dict interface
                    node_i = self._idx_to_node[i]
                    node_j = self._idx_to_node[j]
                    if node_i not in self._dis:
                        self._dis[node_i] = {}
                        self._point[node_i] = {}
                    if node_j not in self._dis:
                        self._dis[node_j] = {}
                        self._point[node_j] = {}

                    self._dis[node_i][node_j] = float(self._dist_matrix[i, j])
                    self._dis[node_j][node_i] = float(self._dist_matrix[i, j])
                    self._point[node_i][node_j] = k
                    self._point[node_j][node_i] = k

        if visit:
            self._visited.add(k)
        else:
            self._retrieved.add(k)

    def distance(self, x, y):
        """Get distance between two nodes"""
        if x == y:
            return 0

        idx_x = self._node_to_idx[x]
        idx_y = self._node_to_idx[y]
        return float(self._dist_matrix[idx_x, idx_y])

    def path(self, x, y):
        """Reconstruct shortest path"""
        if x == y:
            return []

        idx_x = self._node_to_idx[x]
        idx_y = self._node_to_idx[y]

        if self._dist_matrix[idx_x, idx_y] >= 95959590:
            raise Exception

        # Use dict interface for compatibility
        if y in self._point.get(x, {}):
            if self._point[x][y] == "":
                return [y]
            else:
                k = self._point[x][y]
                return self.path(x, k) + self.path(k, y)
        else:
            raise Exception

    def visited(self, k):
        return k in self._visited

    def retrieved(self, k):
        return k in self._visited or k in self._retrieved

    def observed(self, k):
        return k in self._visited or k in self._retrieved or k in self._observed


class CollectiveFloydGraph:
    """Vectorized version of CollectiveFloydGraph with numpy operations"""

    def __init__(self, num_robots, max_nodes=1000):
        # Node indexing
        self._node_to_idx = {}
        self._idx_to_node = {}
        self._node_count = 0
        self._max_nodes = max_nodes

        # Pre-allocate numpy arrays for better performance
        self._dist_matrix = np.full((max_nodes, max_nodes), 95959590, dtype=np.float32)
        np.fill_diagonal(self._dist_matrix, 0)

        # Per-robot visited/observed tracking
        self._visited = [set() for _ in range(num_robots)]
        self._observed = [set() for _ in range(num_robots)]

        # Original dict interface for compatibility
        self._dis = {}
        self._point = {}

    def _get_or_create_idx(self, node):
        """Map node to index, creating new mapping if needed"""
        if node not in self._node_to_idx:
            if self._node_count >= self._max_nodes:
                self._expand_matrices()
            self._node_to_idx[node] = self._node_count
            self._idx_to_node[self._node_count] = node
            self._node_count += 1
        return self._node_to_idx[node]

    def _expand_matrices(self):
        """Expand matrices when we run out of space"""
        new_size = self._max_nodes * 2
        new_dist = np.full((new_size, new_size), 95959590, dtype=np.float32)

        new_dist[:self._max_nodes, :self._max_nodes] = self._dist_matrix
        np.fill_diagonal(new_dist, 0)

        self._dist_matrix = new_dist
        self._max_nodes = new_size

    def distance(self, x, y):
        """Get distance between two nodes - maintains original API"""
        if x == y:
            return 0

        idx_x = self._node_to_idx[x]
        idx_y = self._node_to_idx[y]
        return float(self._dist_matrix[idx_x, idx_y])

    def add_edge(self, x, y, dist):
        """Add edge to graph - maintains original API"""
        # Update matrix representation
        idx_x = self._get_or_create_idx(x)
        idx_y = self._get_or_create_idx(y)

        if dist < self._dist_matrix[idx_x, idx_y]:
            self._dist_matrix[idx_x, idx_y] = dist
            self._dist_matrix[idx_y, idx_x] = dist

            # Update dict interface
            if x not in self._dis:
                self._dis[x] = {}
                self._point[x] = {}
            if y not in self._dis:
                self._dis[y] = {}
                self._point[y] = {}

            self._dis[x][y] = dist
            self._dis[y][x] = dist
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k, robot_id):
        """Vectorized Floyd-Warshall update - maintains original API"""
        # Ensure k exists in our indices
        if k not in self._node_to_idx:
            k_idx = self._get_or_create_idx(k)
        else:
            k_idx = self._node_to_idx[k]

        n = self._node_count

        # Extract relevant slices
        dist_slice = self._dist_matrix[:n, :n]

        # Vectorized update using broadcasting
        dist_to_k = dist_slice[:, k_idx:k_idx+1]  # Shape (n, 1)
        dist_from_k = dist_slice[k_idx:k_idx+1, :]  # Shape (1, n)

        # Compute all possible paths through k
        dist_via_k = dist_to_k + dist_from_k  # Broadcasting: (n, 1) + (1, n) = (n, n)

        # Find where going through k is better
        improved = dist_via_k < dist_slice

        # Update distances in matrix
        self._dist_matrix[:n, :n] = np.where(improved, dist_via_k, dist_slice)

        # Update predecessors where improvements were made
        if np.any(improved):
            # Where we improved, the predecessor of j when coming from i through k is k
            improved_indices = np.where(improved)
            for i, j in zip(improved_indices[0], improved_indices[1]):
                if i != j and i != k_idx and j != k_idx:

                    # Update dict interface
                    node_i = self._idx_to_node[i]
                    node_j = self._idx_to_node[j]
                    if node_i not in self._dis:
                        self._dis[node_i] = {}
                        self._point[node_i] = {}
                    if node_j not in self._dis:
                        self._dis[node_j] = {}
                        self._point[node_j] = {}

                    self._dis[node_i][node_j] = float(self._dist_matrix[i, j])
                    self._dis[node_j][node_i] = float(self._dist_matrix[i, j])
                    self._point[node_i][node_j] = k
                    self._point[node_j][node_i] = k
        # Update visited/observed exactly as original
        self._visited[robot_id].add(k)
        for rid in range(len(self._visited)):
            self._observed[rid].add(k)

    def visited(self, k, robot_id):
        """Check if node k was visited by robot_id - maintains original API"""
        return k in self._visited[robot_id]

    def observed(self, k, robot_id):
        """Check if node k was observed by robot_id - maintains original API"""
        return k in self._observed[robot_id] or self.visited(k, robot_id)

    def path(self, x, y):
        """Reconstruct shortest path - maintains original API"""
        if x == y:
            return []

        idx_x = self._node_to_idx[x]
        idx_y = self._node_to_idx[y]

        if self._dist_matrix[idx_x, idx_y] >= 95959590:
            raise Exception

        # Use dict interface for compatibility
        if y in self._point.get(x, {}):
            if self._point[x][y] == "":
                return [y]
            else:
                k = self._point[x][y]
                return self.path(x, k) + self.path(k, y)
        else:
            raise Exception

    def path_within_nodes(self, start, end, steps_nodes):
        if start == end:
            return []

        if not steps_nodes:
            return None

        max_steps = len(steps_nodes)

        # dp[step][node] = (min_distance, path)
        dp = [{} for _ in range(max_steps + 1)]
        dp[0][start] = (0, [])

        for step in range(max_steps):
            if not dp[step]:
                continue

            allowed_nodes = steps_nodes[step]

            for node, (curr_dist, curr_path) in dp[step].items():
                for neighbor in allowed_nodes:
                    # Check dict first
                    if neighbor in self._point[node] and self._point[node][neighbor] == "":
                        # Get edge weight
                        edge_weight = self.distance(node, neighbor)
                        assert edge_weight < 95959590

                        new_dist = curr_dist + edge_weight
                        new_path = curr_path + [neighbor]


                        if (neighbor not in dp[step + 1] or
                            new_dist < dp[step + 1][neighbor][0]):
                            dp[step + 1][neighbor] = (new_dist, new_path)

        best_path = None
        best_dist = float('inf')

        for step in range(1, max_steps + 1):
            if end in dp[step]:
                dist, path = dp[step][end]
                if dist < best_dist:
                    best_dist = dist
                    best_path = path

        return best_path


@torch.inference_mode()
def get_historical_experience_prompt(action_probs, pred_masks, exp_pairs, steps, threshold, gamma, max_pairs, teacher_paths, mastermind, jump_friendly):
    threshold = [threshold * math.pow(gamma, i) for i in range(steps)]
    target_paths = []
    target_similarities = []
    target_traj_ids = []

    similarities = []
    raw_similarities = []
    anchor_probs = []
    can_probs = []
    all_masks = []

    hit_cnt = [[0 for _ in range(5)] for _ in range(len(exp_pairs))]
    random_hit_cnt = [[0 for _ in range(5)] for _ in range(len(exp_pairs))]
    retrieved_path_cnt = [[0 for _ in range(5)] for _ in range(len(exp_pairs))]
    random_retrieved_path_cnt = [[0 for _ in range(5)] for _ in range(len(exp_pairs))]
    total_path_cnt = [[0 for _ in range(5)] for _ in range(len(exp_pairs))]

    for i, (action_prob, exp_pair) in enumerate(zip(action_probs, exp_pairs)):
        for pair in exp_pair:
            anchor_probs.append(action_prob)
            if mastermind:
                all_masks.append(torch.logical_and(pair['mask'], pred_masks[i]))
                can_probs.append(pair['fts'][1])
            else:
                tmp_can_probs = torch.stack([path_node[1][2] for path_node in pair['path'] if path_node[1] is not None][:steps])
                if tmp_can_probs.size(0) < steps:
                    pad_size = steps - tmp_can_probs.size(0)
                    tmp_can_probs = torch.nn.functional.pad(tmp_can_probs, (0, 0, 0, pad_size), value=0.0)

                if jump_friendly:
                    tmp_mask = [True] * tmp_can_probs.size(0)
                else:
                    tmp_mask = []
                    for path_node in pair['path']:
                        if path_node[1] is not None:
                            tmp_mask.append(True)
                        else:
                            break
                tmp_mask = torch.tensor((tmp_mask + [False] * steps)[:steps]).to(pair['mask'].device)
                all_masks.append(torch.logical_and(pair['mask'], tmp_mask))
                can_probs.append(tmp_can_probs)

    if len(anchor_probs):
        anchor_probs = F.normalize(torch.stack(anchor_probs), dim=-1)
        can_probs = F.normalize(torch.stack(can_probs), dim=-1)
        all_masks = torch.stack(all_masks)
        raw_all_similarities = torch.einsum('bni,bni->bn', anchor_probs, can_probs)
        raw_all_similarities = (raw_all_similarities + 1) / 2
        raw_all_similarities[~all_masks] = 0
        all_similarities = raw_all_similarities.cumprod(dim=-1)
    else:
        all_similarities = []
        raw_all_similarities = []
    for i, (action_prob, exp_pair) in enumerate(zip(action_probs, exp_pairs)):
        similarities.append(all_similarities[:len(exp_pair)])
        raw_similarities.append(raw_all_similarities[:len(exp_pair)])
        all_similarities = all_similarities[len(exp_pair):]
        raw_all_similarities = raw_all_similarities[len(exp_pair):]

    for i, exp_pair in enumerate(exp_pairs):
        result_paths = []
        result_similarities = []
        result_traj_ids = []
        selected_indices = set()
        if len(exp_pair) > 0:
            teacher_path = teacher_paths[i]
            raw_similarity = raw_similarities[i]
            similarity = similarities[i]
            for j in range(steps - 1, -1, -1):
                values, indices = torch.topk(similarity[:, j], min(max_pairs, len(similarity)))
                final_indices = [int(index) for (value, index) in zip(values, indices) if value > threshold[j] and int(index) not in selected_indices][:max_pairs-len(result_paths)]
                for index in final_indices:
                    assert index not in selected_indices and len(result_paths) < max_pairs
                    if jump_friendly:
                        path = [path_node for path_node in exp_pair[index]['path'] if path_node[1] is not None][:j + 1]
                    else:
                        path = exp_pair[index]['path'][:j + 1]
                    result_paths.append(path)
                    result_similarities.append(raw_similarity[index, :j + 1])
                    result_traj_ids.append(exp_pair[index]['traj_id'])
                    selected_indices.add(index)

            for pair in exp_pair:
                j = 0
                while j < len(teacher_path) and j < len(pair['path']) and j < steps and pair['path'][j][0] == teacher_path[j]:
                    total_path_cnt[i][j] += 1
                    j += 1

            for pair in result_paths:
                for j, (vp, _) in enumerate(pair):
                    retrieved_path_cnt[i][j] += 1

                j = 0
                while j < len(teacher_path) and j < len(pair) and j < steps and pair[j][0] == teacher_path[j]:
                    hit_cnt[i][j] += 1
                    j += 1

            random.shuffle(exp_pair)
            for pair in exp_pair[:random.randint(0, len(exp_pair))]:
                rand_step = random.randint(0, steps)
                for j in range(min(rand_step, len(pair['path']))):
                    random_retrieved_path_cnt[i][j] += 1

                j = 0
                while j < len(teacher_path) and j < len(pair['path']) and j < rand_step and pair['path'][j][0] == teacher_path[j]:
                    random_hit_cnt[i][j] += 1
                    j += 1

        target_paths.append(result_paths)
        target_similarities.append(result_similarities)
        target_traj_ids.append(result_traj_ids)
    return target_paths, target_similarities, target_traj_ids, hit_cnt, random_hit_cnt, retrieved_path_cnt, random_retrieved_path_cnt, total_path_cnt


class CollectiveGraphMap:
    def __init__(self, start_vps, shared=False):
        self.start_vps = start_vps  # start viewpoint
        self.shared = shared

        # Long-term memory
        self.graph_pool = CollectiveFloydGraph(len(start_vps))
        self.node_positions_pool = {}
        self.node_embeds_pool = {}
        self.node_values_pool = {}

        # Short-term memory (for current episode)
        self.graph = [FloydGraph() for _ in range(len(start_vps))]
        self.node_positions = [{} for _ in range(len(start_vps))]
        self.node_embeds = [{} for _ in range(len(start_vps))]

        # agent experience
        self.experience_pool = defaultdict(list)
        self.node_to_traj_ids = defaultdict(list)

        self.node_stop_scores = [{} for _ in range(len(start_vps))]  # {viewpoint: prob}
        self.node_nav_scores = [{} for _ in range(len(start_vps))]  # {viewpoint: {t: prob}}
        self.node_step_ids = [{} for _ in range(len(start_vps))]

    def reset(self, start_vps, clear=False):
        self.start_vps = start_vps  # start viewpoint

        # Clear short-term memory for new episode
        self.graph = [FloydGraph() for _ in range(len(start_vps))]
        self.node_positions = [{} for _ in range(len(self.start_vps))]
        self.node_embeds = [{} for _ in range(len(self.start_vps))]

        if clear:
            self.graph_pool = CollectiveFloydGraph(len(self.start_vps))
            self.node_positions_pool.clear()
            self.node_embeds_pool.clear()
            self.node_values_pool.clear()
            self.experience_pool.clear()
        else:
            for _observed, _visited in zip(self.graph_pool._observed, self.graph_pool._visited):
                _observed.union(_visited)
                _visited.clear()
            for vp in self.node_embeds_pool.keys():
                if self.node_embeds_pool[vp][0].requires_grad:
                    self.node_embeds_pool[vp][0] = self.node_embeds_pool[vp][0].detach()
            for vp in self.node_values_pool.keys():
                if self.node_values_pool[vp] is not None and self.node_values_pool[vp].requires_grad:
                    self.node_values_pool[vp] = self.node_values_pool[vp].detach()

        # Reset episode-specific attributes
        self.node_stop_scores = [{} for _ in range(len(self.start_vps))]
        self.node_nav_scores = [{} for _ in range(len(self.start_vps))]
        self.node_step_ids = [{} for _ in range(len(self.start_vps))]

    def visited(self, vp, robot_id):
        return self.graph[robot_id].visited(vp)

    def retrieved(self, vp, robot_id):
        return self.graph[robot_id].retrieved(vp)

    def observed(self, vp, robot_id):
        return self.graph[robot_id].observed(vp)
    
    def reachable(self, vp, robot_id):
        for neighbor in self.graph[robot_id]._point[vp].keys():
            if self.graph[robot_id]._point[vp][neighbor] == "" and self.visited(neighbor, robot_id):
                return True
        return False

    def path(self, x, y, robot_id):
        if self.shared:
            return self.graph_pool.path(x, y)
        else:
            return self.graph[robot_id].path(x, y)

    def distance(self, x, y, robot_id):
        if self.shared:
            return self.graph_pool.distance(x, y)
        else:
            return self.graph[robot_id].distance(x, y)

    def get_connected_nodes(self, robot_id, full_graph=False):
        connected_nodes = []
        other_nodes = []
        if full_graph:
            for k in self.node_positions_pool.keys():
                if self.graph_pool.distance(self.start_vps[robot_id], k) < 95959590:
                    connected_nodes.append(k)
                else:
                    other_nodes.append(k)
        else:
            for k in self.node_positions[robot_id].keys():
                if self.graph[robot_id].distance(self.start_vps[robot_id], k) < 95959590:
                    connected_nodes.append(k)

        if full_graph:
            return connected_nodes, other_nodes
        else:
            return connected_nodes

    def update_graph(self, ob, robot_id):
        self.node_positions[robot_id][ob['viewpoint']] = ob['position']
        self.node_positions_pool[ob['viewpoint']] = ob['position']
        for cc in ob['candidate']:
            dist = calc_position_distance(ob['position'], cc['position'])
            self.node_positions[robot_id][cc['viewpointId']] = cc['position']
            self.graph[robot_id].add_edge(ob['viewpoint'], cc['viewpointId'], dist)
            self.node_positions_pool[cc['viewpointId']] = cc['position']
            self.graph_pool.add_edge(ob['viewpoint'], cc['viewpointId'], dist)
        self.graph[robot_id].update(ob['viewpoint'])
        self.graph_pool.update(ob['viewpoint'], robot_id=robot_id)

    def update_node_embed(self, vp, embed, robot_id, rewrite, value=None, view_id=None, redundant_view=False, observe_friendly=False):
        if rewrite:
            if not self.shared:
                self.node_embeds[robot_id][vp] = [embed, 0]
            self.node_embeds_pool[vp] = [embed, 0]
            self.node_values_pool[vp] = value
        else:
            if not self.shared:
                if vp in self.node_embeds[robot_id]:
                    if self.node_embeds[robot_id][vp][1] > 0:
                        if view_id not in self.node_embeds[robot_id][vp][2] or redundant_view:
                            self.node_embeds[robot_id][vp][0] = self.node_embeds[robot_id][vp][0] + embed
                            self.node_embeds[robot_id][vp][1] += 1
                            self.node_embeds[robot_id][vp][2].append(view_id)
                else:
                    self.node_embeds[robot_id][vp] = [embed, 1, [view_id]]
            if vp in self.node_embeds_pool:
                if self.node_embeds_pool[vp][1] > 0:
                    if view_id not in self.node_embeds_pool[vp][2] or redundant_view:
                        self.node_embeds_pool[vp][0] = self.node_embeds_pool[vp][0] + embed
                        self.node_embeds_pool[vp][1] += 1
                        self.node_embeds_pool[vp][2].append(view_id)
                elif self.shared and observe_friendly:
                    self.graph[robot_id]._observed.add(vp)
            else:
                self.node_embeds_pool[vp] = [embed, 1, [view_id]]

    def get_node_embed(self, vp, robot_id, full_graph=False):
        if self.shared or full_graph:
            if self.node_embeds_pool[vp][1] == 0:
                return self.node_embeds_pool[vp][0]
            else:
                return self.node_embeds_pool[vp][0] / self.node_embeds_pool[vp][1]
        else:
            if self.node_embeds[robot_id][vp][1] == 0:
                return self.node_embeds[robot_id][vp][0]
            else:
                return self.node_embeds[robot_id][vp][0] / self.node_embeds[robot_id][vp][1]

    def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, robot_id, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation), line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.node_positions_pool[cur_vp], self.node_positions_pool[vp],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                if not self.shared:
                    rel_dists.append(
                        [rel_dist / MAX_DIST, self.graph[robot_id].distance(cur_vp, vp) / MAX_DIST,
                         len(self.graph[robot_id].path(cur_vp, vp)) / MAX_STEP]
                    )
                else:
                    rel_dists.append(
                        [rel_dist / MAX_DIST, self.graph_pool.distance(cur_vp, vp) / MAX_DIST,
                         len(self.graph_pool.path(cur_vp, vp)) / MAX_STEP]
                    )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    @torch.inference_mode()
    def smart_search(self, vps, fut_obs, masks, min_beam_size, max_beam_size, robot_ids, filter_rate, dropout_prob, replace, gamma):
        batch_size = len(vps)
        check_neighbours = []
        look_ahead_steps = masks.sum(dim=-1)
        final_nodes = [[set() for _ in range(look_ahead_steps[i])] for i in range(batch_size)]
        for i, vp in enumerate(vps):
            check_neighbours.extend([key for key in self.graph_pool._point[vp].keys() if self.graph_pool._point[vp][
                key] == "" and key in self.node_values_pool and not self.visited(key, robot_ids[i])])
        if not look_ahead_steps.any() or not len(check_neighbours):
            return final_nodes

        steps_nodes = [[set() for __ in range(5)] for _ in range(batch_size)]
        all_nodes = [vps[i] for i in range(batch_size)]
        for i in range(batch_size):
            cans = []
            tmp_cans = [vps[i]]
            for j in range(look_ahead_steps[i]):
                new_cans = []
                for can in tmp_cans:
                    neighbours = [key for key in self.graph_pool._point[can].keys() if self.graph_pool._point[can][
                        key] == "" and key in self.node_values_pool and (key not in cans)]
                    new_cans.extend(neighbours)
                steps_nodes[i][j] = set(new_cans)
                tmp_cans = new_cans
                cans.extend(new_cans)
            all_nodes.extend(cans)
        all_nodes = list(set(all_nodes))
        fut_logits_batch, node2idx = self.compute_logits(fut_obs, all_nodes)
        fut_logits_batch = fut_logits_batch.cpu().numpy()

        for i in range(batch_size):
            for j in range(look_ahead_steps[i]):
                step_nodes = [node for node in steps_nodes[i][j] if not self.visited(node, robot_ids[i])]
                step_filter_rate = filter_rate * math.pow(gamma, j)
                step_logits = fut_logits_batch[i][j]

                total_nodes = len(step_nodes)
                filter_keep = total_nodes - int(total_nodes * step_filter_rate)
                beam_keep = min(max(filter_keep, min_beam_size), max_beam_size)

                sorted_nodes = sorted(step_nodes, key=lambda x: step_logits[node2idx[x]])
                top_nodes = sorted_nodes[-beam_keep:]
                if beam_keep == 0:
                    top_nodes = []
                if total_nodes > min_beam_size:
                    keep_mask = np.random.random(len(top_nodes)) > dropout_prob
                    kept_after_dropout = [node for node, keep in zip(top_nodes, keep_mask) if keep]

                    if len(kept_after_dropout) < beam_keep and replace:
                        raise Exception
                        all_candidates = set(step_nodes)
                        already_kept = set(kept_after_dropout)
                        remaining_candidates = list(all_candidates - already_kept)

                        needed = beam_keep - len(kept_after_dropout)

                        if len(remaining_candidates) >= needed:
                            additional_nodes = np.random.choice(remaining_candidates, needed, replace=False)
                            top_nodes = kept_after_dropout + additional_nodes.tolist()
                        else:
                            top_nodes = kept_after_dropout + remaining_candidates
                    else:
                        top_nodes = kept_after_dropout

                for node in top_nodes:
                    if node not in final_nodes[i][j]:
                        node_path = self.graph_pool.path_within_nodes(vps[i], node, steps_nodes[i][:j+1])
                        for k, n in enumerate(node_path):
                            final_nodes[i][k].add(n)

        return final_nodes

    # todo: randomly chunk beam size
    def random_search(self, vps, look_ahead_steps, min_beam_size, max_beam_size, robot_ids, filter_rate, gamma):
        batch_size = len(vps)
        check_neighbours = []
        look_ahead_steps = np.array([random.randint(0, look_ahead_steps) for i in range(batch_size)])
        final_nodes = [[set() for _ in range(look_ahead_steps[i])] for i in range(batch_size)]
        for i, vp in enumerate(vps):
            check_neighbours.extend([key for key in self.graph_pool._point[vp].keys() if self.graph_pool._point[vp][
                key] == "" and key in self.node_values_pool and not self.visited(key, robot_ids[i])])
        if not look_ahead_steps.any() or not len(check_neighbours):
            return final_nodes

        # bfs
        steps_nodes = [[set() for __ in range(5)] for _ in range(batch_size)]
        for i in range(batch_size):
            cans = []
            tmp_cans = [vps[i]]
            for j in range(look_ahead_steps[i]):
                new_cans = []
                for can in tmp_cans:
                    neighbours = [key for key in self.graph_pool._point[can].keys() if self.graph_pool._point[can][
                        key] == "" and key in self.node_values_pool and (key not in cans)]
                    new_cans.extend(neighbours)
                steps_nodes[i][j] = set(new_cans)
                tmp_cans = new_cans
                cans.extend(new_cans)

        for i in range(batch_size):
            for j in range(look_ahead_steps[i]):
                step_nodes = [node for node in steps_nodes[i][j] if not self.visited(node, robot_ids[i])]
                step_filter_rate = filter_rate * math.pow(gamma, j)
                sorted_nodes = step_nodes
                random.shuffle(sorted_nodes)

                total_nodes = len(sorted_nodes)
                filter_keep = total_nodes - int(total_nodes * step_filter_rate)  # 基于过滤率
                beam_keep = min(max(filter_keep, min_beam_size), max_beam_size)  # 应用beam size约束
                kept_nodes = sorted_nodes[-beam_keep:] if beam_keep > 0 else []

                for node in kept_nodes:
                    if node not in final_nodes[i][j]:
                        node_path = self.graph_pool.path_within_nodes(vps[i], node, steps_nodes[i][:j+1])
                        for k, n in enumerate(node_path):
                            final_nodes[i][k].add(n)

        return final_nodes

    def all_search(self, vp, look_ahead_steps, robot_id):
        final_nodes = [set() for _ in range(look_ahead_steps)]
        check_neighbours = [key for key in self.graph_pool._point[vp].keys() if self.graph_pool._point[vp][
            key] == "" and key in self.node_values_pool and not self.visited(key, robot_id)]
        if not len(check_neighbours):
            return final_nodes

        # bfs
        steps_nodes = [set() for __ in range(5)]
        cans = []
        tmp_cans = [vp]
        for j in range(look_ahead_steps):
            new_cans = []
            for can in tmp_cans:
                neighbours = [key for key in self.graph_pool._point[can].keys() if self.graph_pool._point[can][
                    key] == "" and key in self.node_values_pool and (key not in cans)]
                new_cans.extend(neighbours)
            steps_nodes[j] = set(new_cans)
            tmp_cans = new_cans
            cans.extend(new_cans)
            for node in new_cans:
                if not self.visited(node, robot_id):
                    final_nodes[j].add(node)

        return final_nodes

    @torch.inference_mode()
    def compute_logits(self, fut_obs, nodes):
        batch_size, look_ahead_steps, _ = fut_obs.size()
        node2idx = {node: i for i, node in enumerate(nodes)}
        all_node_values = torch.stack([self.node_values_pool[key] for key in nodes])
        fut_obs = fut_obs.view(batch_size * look_ahead_steps, -1)
        fut_obs = F.normalize(fut_obs, dim=-1)
        all_node_values = F.normalize(all_node_values, dim=-1)
        fut_logits = torch.matmul(fut_obs, all_node_values.t())
        fut_logits = (fut_logits + 1) / 2
        return fut_logits.view(batch_size, look_ahead_steps, -1), node2idx

    def add_experience(self, vp, traj_id, features, mask, aux_path=None):
        if features is not None:
            self.experience_pool[traj_id].append((vp, [feature for feature in features], mask.detach()))
        else:
            self.experience_pool[traj_id].append((vp, None, None))
        for v in aux_path:
            self.experience_pool[traj_id].append((v, None, None))
        self.node_to_traj_ids[vp].append(traj_id)

    def get_experience(self, vp, traj_id):
        pairs = []
        traj_ids = self.node_to_traj_ids[vp]
        for traj in traj_ids:
            if traj != traj_id:
                data = self.experience_pool[traj]
                for j, (anchor_node, feature, mask) in enumerate(data):
                    if anchor_node == vp and feature is not None:
                        idx = j
                        if len(data) - idx > 1:
                            pairs.append({'fts': feature, 'mask': mask, 'path': [(v, f) for v, f, _ in data[idx+1:]], 'traj_id': traj})
        return pairs

    def to_short_term(self, path, robot_id, include_neighbours=False):
        if isinstance(path, list):
            full_path = []
            for i in range(0, len(path) - 1):
                full_path += self.graph_pool.path(path[i], path[i + 1])
        else:
            full_path = path
        for node in full_path:
            if not self.retrieved(node, robot_id):
                neighbours = [key for key in self.graph_pool._point[node].keys() if
                              self.graph_pool._point[node][key] == ""]
                connected = False

                for neighbour in neighbours:
                    self.graph[robot_id].add_edge(node, neighbour, self.graph_pool.distance(node, neighbour))
                    if neighbour in self.node_positions[robot_id]:
                        if self.retrieved(neighbour, robot_id):
                            connected = True
                        if self.node_embeds_pool[neighbour][1] == 0:
                            self.graph[robot_id]._observed.add(neighbour)

                        if not self.shared and self.node_embeds[robot_id][neighbour][1] > 0:
                            if self.node_embeds_pool[neighbour][1] == 0:
                                self.node_embeds[robot_id][neighbour] = self.node_embeds_pool[neighbour]
                            elif len(self.node_embeds_pool[neighbour][2]) > len(
                                    self.node_embeds[robot_id][neighbour][2]):
                                self.node_embeds[robot_id][neighbour] = self.node_embeds_pool[neighbour]

                    elif include_neighbours:
                        self.node_positions[robot_id][neighbour] = self.node_positions_pool[neighbour]
                        if self.node_embeds_pool[neighbour][1] == 0:
                            self.graph[robot_id]._observed.add(neighbour)

                        if not self.shared:
                            self.node_embeds[robot_id][neighbour] = self.node_embeds_pool[neighbour]

                assert connected
                self.graph[robot_id].update(node, visit=False)
                self.node_positions[robot_id][node] = self.node_positions_pool[node]
                if not self.shared:
                    self.node_embeds[robot_id][node] = self.node_embeds_pool[node]
        return

    def get_available_nodes(self, vp, robot_id, look_ahead_steps):
        available_nodes = []
        tmp_cans = [vp]
        for j in range(look_ahead_steps):
            new_cans = []
            for can in tmp_cans:
                neighbours = [key for key in self.graph_pool._point[can].keys() if self.graph_pool._point[can][
                    key] == "" and key in self.node_values_pool and not self.visited(key, robot_id)]
                new_cans.extend(neighbours)
            tmp_cans = new_cans
            available_nodes.extend(new_cans)
        available_nodes = list(set(available_nodes))
        return available_nodes

    def get_status(self, vp, robot_id):
        # if not self.shared:
        graph = self.graph[robot_id]
        visited_nodes = graph._visited - {vp}
        retrieved_nodes = graph._retrieved - graph._visited - {vp}
        prompted_nodes = graph._observed - graph._retrieved - graph._visited - {vp}
        unvisited_nodes = set(self.node_positions[robot_id].keys()) - visited_nodes - retrieved_nodes - prompted_nodes - {vp}
        full_nodes = set(key for key in self.node_positions[robot_id].keys() if self.node_embeds_pool[key][1] == 0)
        partial_nodes = set(key for key in self.node_positions[robot_id].keys() if self.node_embeds_pool[key][1] > 0)
        available_nodes = self.get_available_nodes(vp, robot_id, 5)
        available_nodes = set(available_nodes) - retrieved_nodes - prompted_nodes
        info = {
            'vp': vp,
            'visited': list(visited_nodes),
            'retrieved': list(retrieved_nodes),
            'prompted': list(prompted_nodes),
            'unvisited': list(unvisited_nodes),
            'available': list(available_nodes),
            'full_nodes': list(full_nodes),
            'partial_nodes': list(partial_nodes)
        }
        # else:
        #     visited_nodes = self.graph_pool._visited[robot_id]
        #     prompted_nodes = self.graph_pool._observed[robot_id]
        #     unvisited_nodes = set(self.node_positions_pool.keys()) - visited_nodes - prompted_nodes
        #     info = {
        #         'vp': vp,
        #         'visited': list(visited_nodes),
        #         'prompted': list(prompted_nodes),
        #         'unvisited': list(unvisited_nodes),
        #         'available': []
        #     }

        return info

    def create_episode_view(self, start_vps):
        """Create a lightweight view that shares pool data but has independent episode state"""
        episode_view = CollectiveGraphMap.__new__(CollectiveGraphMap)

        # Share the heavy pool data (read-only)
        episode_view.node_embeds_pool = self.node_embeds_pool      # Shared reference
        episode_view.node_positions_pool = self.node_positions_pool  # Shared reference
        episode_view.node_values_pool = self.node_values_pool      # Shared reference
        episode_view.experience_pool = self.experience_pool        # Shared reference
        episode_view.graph_pool = self.graph_pool                  # Shared reference
        episode_view.shared = self.shared
        episode_view.node_to_traj_ids = self.node_to_traj_ids

        # Create fresh episode-specific state (lightweight)
        episode_view.start_vps = start_vps
        episode_view.graph = [FloydGraph() for _ in range(len(start_vps))]
        episode_view.node_positions = [{} for _ in range(len(start_vps))]
        episode_view.node_embeds = [{} for _ in range(len(start_vps))]
        episode_view.node_stop_scores = [{} for _ in range(len(start_vps))]
        episode_view.node_nav_scores = [{} for _ in range(len(start_vps))]
        episode_view.node_step_ids = [{} for _ in range(len(start_vps))]

        return episode_view
