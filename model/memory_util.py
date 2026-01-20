import math
import numpy as np
import torch
from typing import Optional


def _spatial_decay_bias(mem_pos, query_pos, sigma, lambda_pos, eps, device, dtype, length):
    if mem_pos is None or query_pos is None or lambda_pos == 0:
        return None

    mem_pos = torch.as_tensor(mem_pos, device=device, dtype=dtype).flatten()
    query_pos = torch.as_tensor(query_pos, device=device, dtype=dtype)

    if mem_pos.numel() != length:
        raise ValueError(f'mem_pos length {mem_pos.numel()} does not match memory length {length}')

    sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype)
    sigma_t = torch.clamp(sigma_t, min=eps)
    alpha = torch.exp(-torch.abs(mem_pos - query_pos) / sigma_t)
    bias = lambda_pos * torch.log(alpha + eps)

    return bias.view(1, -1, 1)


def get_similarity(mk, ms, qk, qe, mem_pos=None, query_pos=None, sigma=8.0, lambda_pos=1.0, eps=1e-6):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    if qe is not None:
        # See appendix for derivation
        # or you can just trust me ヽ(ー_ー )ノ
        mk = mk.transpose(1, 2)
        a_sq = (mk.pow(2) @ qe)
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = (-a_sq+two_ab-b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq+two_ab)

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)   # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)   # B*N*HW

    bias = _spatial_decay_bias(
        mem_pos,
        query_pos,
        sigma,
        lambda_pos,
        eps,
        similarity.device,
        similarity.dtype,
        similarity.shape[1],
    )
    if bias is not None:
        similarity = similarity + bias

    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp) # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity

def get_affinity(mk, ms, qk, qe, mem_pos=None, query_pos=None, sigma=8.0, lambda_pos=1.0, eps=1e-6):
    # shorthand used in training with no top-k
    similarity = get_similarity(
        mk,
        ms,
        qk,
        qe,
        mem_pos=mem_pos,
        query_pos=query_pos,
        sigma=sigma,
        lambda_pos=lambda_pos,
        eps=eps,
    )
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity, mv):
    B, CV, T, H, W = mv.shape

    mo = mv.view(B, CV, T*H*W) 
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)

    return mem
