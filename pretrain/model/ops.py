import torch
import torch.nn.functional as F

from .transformer import TransformerEncoder, TransformerEncoderLayer

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


def truncate_text_into_chunks(txt_embeds, txt_lens, num_chunks=4):
    """
    Truncate text embeddings into equal-sized chunks.

    Args:
        txt_embeds: Tensor of shape (batch_size, max_seq_len, embed_dim)
        txt_lens: List or tensor of actual text lengths for each item in the batch
        num_chunks: Number of chunks to divide each sequence into

    Returns:
        Tensor of shape (batch_size, num_chunks, chunk_size, embed_dim)
    """
    batch_size, max_seq_len, embed_dim = txt_embeds.shape

    # Calculate chunk sizes for each sequence in the batch
    chunk_sizes = [max(1, length // num_chunks) for length in txt_lens]
    max_chunk_size = max(chunk_sizes)

    # Initialize output tensor
    output = torch.zeros(batch_size, num_chunks, max_chunk_size, embed_dim,
                         device=txt_embeds.device, dtype=txt_embeds.dtype)

    # Fill output tensor with truncated chunks
    for b in range(batch_size):
        length = txt_lens[b]
        chunk_size = max(1, length // num_chunks)

        for c in range(num_chunks):
            if c != 0:
                start_idx = c * chunk_size
            else:
                start_idx = 1
            # Handle the case where the last chunk might be smaller
            end_idx = min((c + 1) * chunk_size, length)

            # Extract the chunk
            chunk = txt_embeds[b, start_idx:end_idx]

            # Pad if necessary (for the last chunk which might be smaller)
            if chunk.shape[0] < max_chunk_size:
                padded_chunk = torch.zeros(max_chunk_size, embed_dim,
                                           device=chunk.device, dtype=chunk.dtype)
                padded_chunk[:chunk.shape[0]] = chunk
                chunk = padded_chunk

            # Store the chunk
            output[b, c, :chunk.shape[0]] = chunk

    return output



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
                [tensors[i], torch.zeros([max_len-lens[i]]+hid, dtype=dtype).to(device)],
                dim=0
            )
        else:
            tmp = tensors[i]
        output.append(tmp)
    output = torch.stack(output, 0)
    return output
