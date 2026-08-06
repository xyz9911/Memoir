import numpy as np
import torch
import sys
from typing import Dict, Iterable, List, Optional, Tuple, Union
from gym import spaces
from torch.nn import functional as F
import warnings
from reprlib import recursive_repr
from utils.transformer import TransformerEncoder, TransformerEncoderLayer

BertLayerNorm = torch.nn.LayerNorm
import random

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')

def zdistr(pp, stoch_discrete, stoch_dim):
    # pp = post or prior
    if stoch_discrete:
        logits = pp.reshape(pp.shape[:-1] + (stoch_dim, stoch_discrete))
        distr = torch.distributions.OneHotCategoricalStraightThrough(logits=logits.float())  # NOTE: .float() needed to force float32 on AMP  # This makes d.entropy() and d.kl() sum over stoch_dim
        return distr
    else:
        mean, std = torch.chunk(pp, 2, dim=-1)
        std = F.softplus(std) + 0.1
        distr = torch.distributions.Normal(mean, std)
        return distr


def find_best_adherence(scan, full_path, gt_path, scanvp_cands):
    chunks = []
    cursor = 0
    while cursor < len(full_path):
        chunk = []
        cursor_gt = gt_path.index(full_path[cursor])
        if cursor > 0 and full_path[cursor] in scanvp_cands['%s_%s' % (scan, full_path[cursor - 1])]:
            chunk.append(cursor - 1)
        while cursor < len(full_path) and cursor_gt < len(gt_path) and full_path[cursor] == gt_path[cursor_gt]:
            chunk.append(cursor)
            cursor += 1
            cursor_gt += 1
        if len(chunk) > 1:
            if cursor == len(full_path) and cursor_gt == len(gt_path):
                chunks.append(chunk)
            elif len(chunk) > 2:
                chunks.append(chunk[:-1])
        while cursor < len(full_path) and full_path[cursor] not in gt_path[cursor_gt:]:
            cursor += 1
    return chunks


def construct_td_pairs(full_beliefs, full_paths, gt_paths, gt_act_labels, gt_img_embeds, scans, scanvp_cands):
    t1_beliefs, t2_beliefs, dts = [], [], []
    t2_act_labels, t2_gt_img_embeds = [], []

    for i, full_path in enumerate(full_paths):
        gt_path = gt_paths[i]
        chunks = find_best_adherence(scans[i], full_path, gt_path, scanvp_cands)
        for chunk in chunks:
            anchor = random.choice(chunk[:-1])
            t2_idx = random.choice([x for x in chunk if x > anchor])
            for dt in range(5):
                if t2_idx - dt - 1 >= chunk[0]:
                    t1_beliefs.append(full_beliefs[i, t2_idx - dt - 1])
                    t2_beliefs.append(full_beliefs[i, t2_idx])
                    dts.append(dt)
                    t2_act_labels.append(gt_act_labels[i][t2_idx])
                    t2_gt_img_embeds.append(gt_img_embeds[i][t2_idx])

    # Convert lists to tensors
    t1_beliefs = torch.stack(t1_beliefs) if t1_beliefs else []
    t2_beliefs = torch.stack(t2_beliefs) if t2_beliefs else []
    dts = torch.LongTensor(dts).cuda() if dts else []
    t2_act_labels = torch.stack(t2_act_labels) if t2_act_labels else []
    t2_gt_img_embeds = torch.stack(t2_gt_img_embeds) if t2_gt_img_embeds else []
    return t1_beliefs, t2_beliefs, dts, t2_act_labels, t2_gt_img_embeds


def info_nce_loss(z, x, labels=None, temperature=0.05, reduction='mean', normalize=True):
    if normalize:
        z = F.normalize(z, dim=1)
        x = F.normalize(x, dim=1)
    logits = torch.matmul(z, x.t())
    if labels is None:
        labels = torch.arange(len(z), device=z.device)
    info_nce = F.cross_entropy(logits / temperature, labels, reduction=reduction)
    return info_nce


def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    device = tensors[0].device
    output = torch.zeros(*size, dtype=dtype).to(device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)

    if isinstance(seq_lens, torch.Tensor):
        device = seq_lens.device
        masks = torch.arange(max_len).to(device).repeat(len(seq_lens), 1) < seq_lens.unsqueeze(1)
        return masks

    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=bool)

    seq_lens = np.array(seq_lens)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks


def length2mask(length, size=None, device='cpu'):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64, device=device).unsqueeze(0).repeat(batch_size, 1)
            > (torch.LongTensor(length) - 1).unsqueeze(1).to(device))
    return mask


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def obs_as_tensor(
        obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: torch.device
):
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return torch.from_numpy(obs.copy()).float().to(device)
    elif isinstance(obs, dict):
        return {key: torch.from_numpy(_obs.copy()).float().to(device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


def preprocess_batch_dim(obs: Dict[Union[str, int], torch.Tensor]):
    for key, value in obs.items():
        shape = tuple(value.size())
        batch_size, robot_num = shape[:2]
        obs[key] = value.view((batch_size * robot_num,) + shape[2:])
    return obs


def preprocess_obs(
        obs: np.ndarray,
        observation_space: spaces.Space,
        device: torch.device,
        normalize_images: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.

    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        obs = obs_as_tensor(obs, device)
        if normalize_images:
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key], device,
                                                   normalize_images=normalize_images)
        return preprocessed_obs

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")


class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*self.args, *args, **keywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
                                          self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
                (kwds is not None and not isinstance(kwds, dict)) or
                (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args)  # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict:  # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds


try:
    from _functools import partial
except ImportError:
    pass


def create_transformer_encoder(config, num_layers, norm=False):
    enc_layer = TransformerEncoderLayer(
        config.hidden_size, config.num_attention_heads,
        dim_feedforward=config.intermediate_size,
        dropout=config.hidden_dropout_prob,
        activation=config.hidden_act,
        normalize_before=True
    )
    if norm:
        norm_layer = BertLayerNorm(config.hidden_size, eps=1e-12)
    else:
        norm_layer = None
    return TransformerEncoder(enc_layer, num_layers, norm=norm_layer, batch_first=True)


def extend_neg_masks(masks, dtype=None):
    """
    mask from (N, L) into (N, 1(H), 1(L), L) and make it negative
    """
    if dtype is None:
        dtype = torch.float
    extended_masks = masks.unsqueeze(1).unsqueeze(2)
    extended_masks = extended_masks.to(dtype=dtype)
    extended_masks = (1.0 - extended_masks) * -10000.0
    return extended_masks


def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)
    batch_size = len(seq_lens)
    device = seq_lens.device

    masks = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    masks = masks < seq_lens.unsqueeze(1)
    return masks


def pad_tensors_wgrad(tensors, lens=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    batch_size = len(tensors)
    hid = list(tensors[0].size()[1:])

    device = tensors[0].device
    dtype = tensors[0].dtype

    output = []
    for i in range(batch_size):
        if lens[i] < max_len:
            tmp = torch.cat(
                [tensors[i], torch.zeros([max_len - lens[i]] + hid, dtype=dtype).to(device)],
                dim=0
            )
        else:
            tmp = tensors[i]
        output.append(tmp)
    output = torch.stack(output, 0)
    return output
