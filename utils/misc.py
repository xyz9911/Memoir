import os
import random
import numpy as np
import torch
import json
import networkx as nx
from numpy.linalg import norm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed import init_distributed
from .logger import LOGGER

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def ndtw_initialize(scans, connectivity_dir):
    ndtw_criterion = {}
    for scan in scans:
        # print('path_scan_id', path_scan_id)
        ndtw_graph = ndtw_graphload(scan, connectivity_dir)
        ndtw_criterion[scan] = DTW(ndtw_graph)
    return ndtw_criterion

def ndtw_graphload(scan, connectivity_dir):
    """Loads a networkx graph for a given scan.
    Args:
    connections_file: A string with the path to the .json file with the
      connectivity_old information.
    Returns:
    A networkx graph.
    """
    connections_file = os.path.join(connectivity_dir, '{}_connectivity.json').format(scan)
    with open(connections_file) as f:
        lines = json.load(f)
        nodes = np.array([x['image_id'] for x in lines])
        matrix = np.array([x['unobstructed'] for x in lines])
        mask = np.array([x['included'] for x in lines])

        matrix = matrix[mask][:, mask]
        nodes = nodes[mask]

        pos2d = {x['image_id']: np.array(x['pose'])[[3, 7]] for x in lines}
        pos3d = {x['image_id']: np.array(x['pose'])[[3, 7, 11]] for x in lines}

    graph = nx.from_numpy_array(matrix)
    graph = nx.relabel.relabel_nodes(graph, dict(enumerate(nodes)))
    nx.set_node_attributes(graph, pos2d, 'pos2d')
    nx.set_node_attributes(graph, pos3d, 'pos3d')

    weight2d = {(u, v): norm(pos2d[u] - pos2d[v]) for u, v in graph.edges}
    weight3d = {(u, v): norm(pos3d[u] - pos3d[v]) for u, v in graph.edges}
    nx.set_edge_attributes(graph, weight2d, 'weight2d')
    nx.set_edge_attributes(graph, weight3d, 'weight3d')

    return graph

class DTW(object):
  """Dynamic Time Warping (DTW) evaluation metrics.
  Python doctest:
  >>> graph = nx.grid_graph([3, 4])
  >>> prediction = [(0, 0), (1, 0), (2, 0), (3, 0)]
  >>> reference = [(0, 0), (1, 0), (2, 1), (3, 2)]
  >>> dtw = DTW(graph)
  >>> assert np.isclose(dtw(prediction, reference, 'dtw'), 3.0)
  >>> assert np.isclose(dtw(prediction, reference, 'ndtw'), 0.77880078307140488)
  >>> assert np.isclose(dtw(prediction, reference, 'sdtw'), 0.77880078307140488)
  >>> assert np.isclose(dtw(prediction[:2], reference, 'sdtw'), 0.0)
  """

  def __init__(self, graph, weight='weight', threshold=3.0):
    """Initializes a DTW object.
    Args:
      graph: networkx graph for the environment.
      weight: networkx edge weight key (str).
      threshold: distance threshold $d_{th}$ (float).
    """
    self.graph = graph
    self.weight = weight
    self.threshold = threshold
    self.distance = dict(
        nx.all_pairs_dijkstra_path_length(self.graph, weight=self.weight))

  def __call__(self, prediction, reference, metric='sdtw'):
    """Computes DTW metrics.
    Args:
      prediction: list of nodes (str), path predicted by agent.
      reference: list of nodes (str), the ground truth path.
      metric: one of ['ndtw', 'sdtw', 'dtw'].
    Returns:
      the DTW between the prediction and reference path (float).
    """
    assert metric in ['ndtw', 'sdtw', 'dtw']

    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
      for j in range(1, len(reference)+1):
        best_previous_cost = min(
            dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
        cost = self.distance[prediction[i-1]][reference[j-1]]
        dtw_matrix[i][j] = cost + best_previous_cost
    dtw = dtw_matrix[len(prediction)][len(reference)]

    if metric == 'dtw':
      return dtw

    ndtw = np.exp(-dtw/(self.threshold * len(reference)))
    if metric == 'ndtw':
      return ndtw

    success = self.distance[prediction[-1]][reference[-1]] <= self.threshold
    return success * ndtw

def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p and name != 'drop_env' and 'prior' not in name:
                module.p = drop_p
                LOGGER.info(f'{name} set to {drop_p}')

def set_cuda(opts):
    """
    Initialize CUDA for distributed computing
    """
    if not torch.cuda.is_available():
        assert opts.local_rank == -1, opts.local_rank
        return True, 0, torch.device("cpu")

    # get device settings
    if opts.local_rank != -1:
        init_distributed(opts)
        torch.cuda.set_device(opts.local_rank)
        device = torch.device("cuda", opts.local_rank)
        n_gpu = 1
        default_gpu = dist.get_rank() == 0
        if default_gpu:
            LOGGER.info(f"Found {dist.get_world_size()} GPUs")
    else:
        default_gpu = True
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()

    return default_gpu, n_gpu, device

def wrap_model(
    model: torch.nn.Module, device: torch.device, local_rank: int
) -> torch.nn.Module:
    model.to(device)

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        # At the time of DDP wrapping, parameters and buffers (i.e., model.state_dict())
        # on rank0 are broadcasted to all other ranks.
    elif torch.cuda.device_count() > 1:
        LOGGER.info("Using data parallel")
        model = torch.nn.DataParallel(model)

    return model


class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return
