# import math
# import os
# import random
# import sys
# import time
# from itertools import count, islice
# from math import cos, gamma, pi, sin, sqrt
# from typing import Callable, Iterator, List
#
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch.distributions import Categorical, kl_divergence
#
#
# class ExponentialMovingAverage(nn.Module):
#     shadow: torch.Tensor
#
#     def __init__(self, initValue, decay):
#         super().__init__()
#         if initValue is None:
#             self.shadow = None
#         else:
#             self.register_buffer("shadow", initValue.clone().detach())
#         self.decay = decay
#
#     def forward(self, x):
#         if self.shadow is None:
#             self.register_buffer("shadow", x.clone().detach())
#             return self.shadow
#         self.shadow.copy_((1 - self.decay) * self.shadow + self.decay * x)
#         return self.shadow
#
#     @property
#     def Value(self):
#         return self.shadow
#
#
# class NetworkExponentialMovingAverage(nn.Module):
#     def __init__(self, network, decay):
#         super().__init__()
#         for (name, weights) in network.named_parameters():
#             self.register_buffer(name.replace('.', ''), weights.clone().detach())
#         self.decay = decay
#
#     def forward(self, network):
#         for (name, weights) in network.named_parameters():
#             shadow = getattr(self, name.replace('.', ''))
#             shadow.copy_((1 - self.decay) * shadow + self.decay * weights)
#             weights.data.copy_(shadow)
#         return network
#
#
# class ExponentialMovingAverage(nn.Module):
#     shadow: torch.Tensor
#     """
#     updated sub prototype codebook
#     """
#
#     def __init__(self, init_value, decay):
#         super().__init__()
#         if init_value is None:
#             self.shadow = None
#         else:
#             self.register_buffer("shadow", init_value.clone().detach())
#         self.decay = decay
#
#     def forward(self, x):
#         if self.shadow is None:
#             self.register_buffer("shadow", x.clone().detach())
#             return self.shadow
#         self.shadow.copy_((1 - self.decay) * self.shadow + self.decay * x)
#         return self.shadow
#
#     @property
#     def Value(self):
#         return self.shadow
#
#
# class NetworkExponentialMovingAverage(nn.Module):
#     def __init__(self, network, decay):
#         super().__init__()
#         for (name, weights) in network.named_parameters():
#             self.register_buffer(name.replace('.', ''), weights.clone().detach())
#         self.decay = decay
#
#     def forward(self, network):
#         for (name, weights) in network.named_parameters():
#             shadow = getattr(self, name.replace('.', ''))
#             shadow.copy_((1 - self.decay) * shadow + self.decay * weights)
#             weights.data.copy_(shadow)
#         return network
#
#
# def cal_kl(distance):
#     """
#     计算 KL 散度
#     Args:
#         distance (Tensor): dist
#         k (Tensor): K value
#
#     Returns:
#         Tensor: kl distance
#     """
#     a = Categorical(logits=distance) #[N,K]
#     b = Categorical(probs=torch.ones([distance.shape[0],distance.shape[1]]).cuda() / distance.shape[1])
#     loss = kl_divergence(a, b).mean()
#
#     return loss
#
#
# # refer to https://github.com/stanis-morozov/unq/blob/e8f7f43699c74be415732d914b01662ce3f60612/lib/quantizer.py#L197
# def gumbel_variance(ips, p, alpha, M, square_cv=True):
#     """
#     计算协方差满足分布
#
#     refer to Unsupervised Neural Quantization for Compressed-Domain Similarity Search
#
#     Args:
#         ips (_type_): _description_
#         p (_type_): _description_
#         alpha (_type_): _description_
#         M (_type_): _description_
#         square_cv (bool, optional): _description_. Defaults to True.
#
#     Returns:
#         _type_: _description_
#     """
#     codes = F.gumbel_softmax(ips / alpha * M, dim=-1)  # gumbel-softmax logits
#     load = torch.mean(p, dim=0)  # [..., codebook_size]
#     mean = load.mean()
#     variance = torch.mean((load - mean) ** 2)
#     if square_cv:
#         counters['cv_squared'] = variance / (mean ** 2 + eps)
#         counters['reg'] += cv_coeff * counters['cv_squared']
#     else:
#         counters['cv'] = torch.sqrt(variance + eps) / (mean + eps)
#         counters['reg'] += cv_coeff * counters['cv']
#
#     return counters
#
#
# # start uniform hypersphere
# def int_sin_m(x: float, m: int) -> float:
#     """
#     Computes the integral of sin^m(t) dt from 0 to x recursively
#     """
#     if m == 0:
#         return x
#     elif m == 1:
#         return 1 - cos(x)
#     else:
#         return (m - 1) / m * int_sin_m(x, m - 2) - cos(x) * sin(x) ** (m - 1) / m
#
#
# def primes() -> Iterator[int]:
#     """
#     Returns an infinite generator of prime numbers
#     """
#     yield from (2, 3, 5, 7)
#     composites = {}
#     ps = primes()
#     next(ps)
#     p = next(ps)
#     assert p == 3
#     psq = p * p
#     for i in count(9, 2):
#         if i in composites:  # composite
#             step = composites.pop(i)
#         elif i < psq:  # prime
#             yield i
#             continue
#         else:  # composite, = p*p
#             assert i == psq
#             step = 2 * p
#             p = next(ps)
#             psq = p * p
#         i += step
#         while i in composites:
#             i += step
#         composites[i] = step
#
#
# def inverse_increasing(
#         func: Callable[[float], float],
#         target: float,
#         lower: float,
#         upper: float,
#         atol: float = 1e-10,
# ) -> float:
#     """
#     Returns func inverse of target between lower and upper
#
#     inverse is accurate to an absolute tolerance of atol, and
#     must be monotonically increasing over the interval lower
#     to upper
#     """
#     mid = (lower + upper) / 2
#     approx = func(mid)
#     while abs(approx - target) > atol:
#         if approx > target:
#             upper = mid
#         else:
#             lower = mid
#         mid = (upper + lower) / 2
#         approx = func(mid)
#     return mid
#
#
# def uniform_hypersphere(d: int, n: int) -> List[List[float]]:  # 初始化的点>>要用的点，上下左右的点距离都比较近
#     """Generate n points over the d dimensional hypersphere"""
#     assert d > 1
#     assert n > 0
#     points = [[1 for _ in range(d)] for _ in range(n)]
#     for i in range(n):
#         t = 2 * pi * i / n
#         points[i][0] *= sin(t)
#         points[i][1] *= cos(t)
#     for dim, prime in zip(range(2, d), primes()):
#         offset = sqrt(prime)
#         mult = gamma(dim / 2 + 0.5) / gamma(dim / 2) / sqrt(pi)
#
#         def dim_func(y):
#             return mult * int_sin_m(y, dim - 1)
#
#         for i in range(n):
#             deg = inverse_increasing(dim_func, i * offset % 1, 0, pi)
#             for j in range(dim):
#                 points[i][j] *= sin(deg)
#             points[i][dim] *= cos(deg)
#     return points
#
#
# # end uniform hypersphere
#
# def _cosine_similarity(codebook):
#     inner_product = codebook @ codebook.T
#     norm = (codebook ** 2).sum(-1).sqrt()
#     return inner_product / (norm[:, None] * norm)
#
#
# def get_semantic_diversity1():
#     freq_dict = {
#         "on": 13,
#         "has": 12,
#         "in": 12,
#         "of": 12,
#         "wearing": 6,
#         "near": 12,
#         "with": 12,
#         "above": 11,
#         "holding": 9,
#         "behind": 11,
#         "under": 12,
#         "sitting on": 9,
#         "wears": 4,
#         "standing on": 7,
#         "in front of": 10,
#         "attached to": 9,
#         "at": 7,
#         "hanging from": 6,
#         "over": 10,
#         "for": 9,
#         "riding": 3,
#         "carrying": 5,
#         "eating": 3,
#         "walking on": 2,
#         "playing": 1,
#         "covering": 6,
#         "laying on": 6,
#         "along": 3,
#         "watching": 3,
#         "and": 10,
#         "between": 8,
#         "belonging to": 3,
#         "painted on": 3,
#         "against": 5,
#         "looking at": 5,
#         "from": 5,
#         "parked on": 1,
#         "to": 6,
#         "made of": 2,
#         "covered in": 3,
#         "mounted on": 3,
#         "says": 1,
#         "part of": 5,
#         "across": 4,
#         "flying in": 1,
#         "using": 3,
#         "on back of": 5,
#         "lying on": 5,
#         "growing on": 1,
#         "walking in": 2,
#     }
#
#     pred2idx = {
#         'and': 5,
#         'says': 39,
#         'belonging to': 9,
#         'over': 33,
#         'parked on': 35,
#         'growing on': 18,
#         'standing on': 41,
#         'made of': 27,
#         'attached to': 7,
#         'at': 6,
#         'in': 22,
#         'hanging from': 19,
#         'wears': 49,
#         'in front of': 23,
#         'from': 17,
#         'for': 16,
#         'watching': 47,
#         'lying on': 26,
#         'to': 42,
#         'behind': 8,
#         'flying in': 15,
#         'looking at': 25,
#         'on back of': 32,
#         'holding': 21,
#         'between': 10,
#         'laying on': 24,
#         'riding': 38,
#         'has': 20,
#         'across': 2,
#         'wearing': 48,
#         'walking on': 46,
#         'eating': 14,
#         'above': 1,
#         'part of': 36,
#         'walking in': 45,
#         'sitting on': 40,
#         'under': 43,
#         'covered in': 12,
#         'carrying': 11,
#         'using': 44,
#         'along': 4,
#         'with': 50,
#         'on': 31,
#         'covering': 13,
#         'of': 30,
#         'against': 3,
#         'playing': 37,
#         'near': 29,
#         'painted on': 34,
#         'mounted on': 28
#     }
#
#     pred_list = [0] * 51
#     pred_list[0] = 1
#     for k, v in freq_dict.items():
#         pred_list[int(pred2idx[k])] = 1
#
#     return pred_list
#
#
# def get_semantic_diversity():
#     freq_dict = {
#         "on": 13,
#         "has": 12,
#         "in": 12,
#         "of": 12,
#         "wearing": 6,
#         "near": 12,
#         "with": 12,
#         "above": 11,
#         "holding": 9,
#         "behind": 11,
#         "under": 12,
#         "sitting on": 9,
#         "wears": 4,
#         "standing on": 7,
#         "in front of": 10,
#         "attached to": 9,
#         "at": 7,
#         "hanging from": 6,
#         "over": 10,
#         "for": 9,
#         "riding": 3,
#         "carrying": 5,
#         "eating": 3,
#         "walking on": 2,
#         "playing": 1,
#         "covering": 6,
#         "laying on": 6,
#         "along": 3,
#         "watching": 3,
#         "and": 10,
#         "between": 8,
#         "belonging to": 3,
#         "painted on": 3,
#         "against": 5,
#         "looking at": 5,
#         "from": 5,
#         "parked on": 1,
#         "to": 6,
#         "made of": 2,
#         "covered in": 3,
#         "mounted on": 3,
#         "says": 1,
#         "part of": 5,
#         "across": 4,
#         "flying in": 1,
#         "using": 3,
#         "on back of": 5,
#         "lying on": 5,
#         "growing on": 1,
#         "walking in": 2,
#     }
#
#     pred2idx = {
#         'and': 5,
#         'says': 39,
#         'belonging to': 9,
#         'over': 33,
#         'parked on': 35,
#         'growing on': 18,
#         'standing on': 41,
#         'made of': 27,
#         'attached to': 7,
#         'at': 6,
#         'in': 22,
#         'hanging from': 19,
#         'wears': 49,
#         'in front of': 23,
#         'from': 17,
#         'for': 16,
#         'watching': 47,
#         'lying on': 26,
#         'to': 42,
#         'behind': 8,
#         'flying in': 15,
#         'looking at': 25,
#         'on back of': 32,
#         'holding': 21,
#         'between': 10,
#         'laying on': 24,
#         'riding': 38,
#         'has': 20,
#         'across': 2,
#         'wearing': 48,
#         'walking on': 46,
#         'eating': 14,
#         'above': 1,
#         'part of': 36,
#         'walking in': 45,
#         'sitting on': 40,
#         'under': 43,
#         'covered in': 12,
#         'carrying': 11,
#         'using': 44,
#         'along': 4,
#         'with': 50,
#         'on': 31,
#         'covering': 13,
#         'of': 30,
#         'against': 3,
#         'playing': 37,
#         'near': 29,
#         'painted on': 34,
#         'mounted on': 28
#     }
#
#     pred_list = [0] * 51
#     pred_list[0] = 1
#     for k, v in freq_dict.items():
#         pred_list[int(pred2idx[k])] = v
#
#     return pred_list
#
#
# def get_sub_proto_label(mode='concat'):
#     semantic_list = get_semantic_diversity()
#     idx2concept = range(sum(semantic_list))
#
#     if mode.lower() != "add" and mode.lower() != "concat" and mode.lower() != "clip":
#         raise ValueError("Incorrect mode you input it.")
#     cluster_dict = torch.load(
#         f"/home/xiejunlin/workspace/Intra-Imbalanced-SGG/datasets/datafiles/intra-work/cluster_results/cluster_dict_{mode}.pt")
#
#     return cluster_dict

import math
import os
import random
import sys
import time
from itertools import count, islice
from math import cos, gamma, pi, sin, sqrt
from typing import Callable, Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, kl_divergence


class ExponentialMovingAverage(nn.Module):
    shadow: torch.Tensor

    def __init__(self, initValue, decay):
        super().__init__()
        if initValue is None:
            self.shadow = None
        else:
            self.register_buffer("shadow", initValue.clone().detach())
        self.decay = decay

    def forward(self, x):
        if self.shadow is None:
            self.register_buffer("shadow", x.clone().detach())
            return self.shadow
        self.shadow.copy_((1 - self.decay) * self.shadow + self.decay * x)
        return self.shadow

    @property
    def Value(self):
        return self.shadow


class NetworkExponentialMovingAverage(nn.Module):
    def __init__(self, network, decay):
        super().__init__()
        for (name, weights) in network.named_parameters():
            self.register_buffer(name.replace('.', ''), weights.clone().detach())
        self.decay = decay

    def forward(self, network):
        for (name, weights) in network.named_parameters():
            shadow = getattr(self, name.replace('.', ''))
            shadow.copy_((1 - self.decay) * shadow + self.decay * weights)
            weights.data.copy_(shadow)
        return network


class ExponentialMovingAverage(nn.Module):
    shadow: torch.Tensor
    """
    updated sub prototype codebook
    """

    def __init__(self, init_value, decay):
        super().__init__()
        if init_value is None:
            self.shadow = None
        else:
            self.register_buffer("shadow", init_value.clone().detach())
        self.decay = decay

    def forward(self, x):
        if self.shadow is None:
            self.register_buffer("shadow", x.clone().detach())
            return self.shadow
        self.shadow.copy_((1 - self.decay) * self.shadow + self.decay * x)
        return self.shadow

    @property
    def Value(self):
        return self.shadow


class NetworkExponentialMovingAverage(nn.Module):
    def __init__(self, network, decay):
        super().__init__()
        for (name, weights) in network.named_parameters():
            self.register_buffer(name.replace('.', ''), weights.clone().detach())
        self.decay = decay

    def forward(self, network):
        for (name, weights) in network.named_parameters():
            shadow = getattr(self, name.replace('.', ''))
            shadow.copy_((1 - self.decay) * shadow + self.decay * weights)
            weights.data.copy_(shadow)
        return network


def cal_kl(distance):
    """
    计算 KL 散度
    Args:
        distance (Tensor): dist
        k (Tensor): K value

    Returns:
        Tensor: kl distance
    """
    a = Categorical(logits=distance) #[N,K]
    b = Categorical(probs=torch.ones([distance.shape[0],distance.shape[1]]).cuda() / distance.shape[1])
    loss = kl_divergence(a, b).mean()

    return loss


# refer to https://github.com/stanis-morozov/unq/blob/e8f7f43699c74be415732d914b01662ce3f60612/lib/quantizer.py#L197
def gumbel_variance(ips, p, alpha, M, square_cv=True):
    """
    计算协方差满足分布

    refer to Unsupervised Neural Quantization for Compressed-Domain Similarity Search

    Args:
        ips (_type_): _description_
        p (_type_): _description_
        alpha (_type_): _description_
        M (_type_): _description_
        square_cv (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    codes = F.gumbel_softmax(ips / alpha * M, dim=-1)  # gumbel-softmax logits
    load = torch.mean(p, dim=0)  # [..., codebook_size]
    mean = load.mean()
    variance = torch.mean((load - mean) ** 2)
    if square_cv:
        counters['cv_squared'] = variance / (mean ** 2 + eps)
        counters['reg'] += cv_coeff * counters['cv_squared']
    else:
        counters['cv'] = torch.sqrt(variance + eps) / (mean + eps)
        counters['reg'] += cv_coeff * counters['cv']

    return counters


# start uniform hypersphere
def int_sin_m(x: float, m: int) -> float:
    """
    Computes the integral of sin^m(t) dt from 0 to x recursively
    """
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - cos(x) * sin(x) ** (m - 1) / m


def primes() -> Iterator[int]:
    """
    Returns an infinite generator of prime numbers
    """
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:  # composite
            step = composites.pop(i)
        elif i < psq:  # prime
            yield i
            continue
        else:  # composite, = p*p
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step


def inverse_increasing(
        func: Callable[[float], float],
        target: float,
        lower: float,
        upper: float,
        atol: float = 1e-10,
) -> float:
    """
    Returns func inverse of target between lower and upper

    inverse is accurate to an absolute tolerance of atol, and
    must be monotonically increasing over the interval lower
    to upper
    """
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid


def uniform_hypersphere(d: int, n: int) -> List[List[float]]:  # 初始化的点>>要用的点，上下左右的点距离都比较近
    """Generate n points over the d dimensional hypersphere"""
    assert d > 1
    assert n > 0
    points = [[1 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        t = 2 * pi * i / n
        points[i][0] *= sin(t)
        points[i][1] *= cos(t)
    for dim, prime in zip(range(2, d), primes()):
        offset = sqrt(prime)
        mult = gamma(dim / 2 + 0.5) / gamma(dim / 2) / sqrt(pi)

        def dim_func(y):
            return mult * int_sin_m(y, dim - 1)

        for i in range(n):
            deg = inverse_increasing(dim_func, i * offset % 1, 0, pi)
            for j in range(dim):
                points[i][j] *= sin(deg)
            points[i][dim] *= cos(deg)
    return points


# end uniform hypersphere

def _cosine_similarity(codebook):
    inner_product = codebook @ codebook.T
    norm = (codebook ** 2).sum(-1).sqrt()
    return inner_product / (norm[:, None] * norm)


def get_semantic_diversity1():
    freq_dict = {
        "on": 13,
        "has": 12,
        "in": 12,
        "of": 12,
        "wearing": 6,
        "near": 12,
        "with": 12,
        "above": 11,
        "holding": 9,
        "behind": 11,
        "under": 12,
        "sitting on": 9,
        "wears": 4,
        "standing on": 7,
        "in front of": 10,
        "attached to": 9,
        "at": 7,
        "hanging from": 6,
        "over": 10,
        "for": 9,
        "riding": 3,
        "carrying": 5,
        "eating": 3,
        "walking on": 2,
        "playing": 1,
        "covering": 6,
        "laying on": 6,
        "along": 3,
        "watching": 3,
        "and": 10,
        "between": 8,
        "belonging to": 3,
        "painted on": 3,
        "against": 5,
        "looking at": 5,
        "from": 5,
        "parked on": 1,
        "to": 6,
        "made of": 2,
        "covered in": 3,
        "mounted on": 3,
        "says": 1,
        "part of": 5,
        "across": 4,
        "flying in": 1,
        "using": 3,
        "on back of": 5,
        "lying on": 5,
        "growing on": 1,
        "walking in": 2,
    }

    pred2idx = {
        'and': 5,
        'says': 39,
        'belonging to': 9,
        'over': 33,
        'parked on': 35,
        'growing on': 18,
        'standing on': 41,
        'made of': 27,
        'attached to': 7,
        'at': 6,
        'in': 22,
        'hanging from': 19,
        'wears': 49,
        'in front of': 23,
        'from': 17,
        'for': 16,
        'watching': 47,
        'lying on': 26,
        'to': 42,
        'behind': 8,
        'flying in': 15,
        'looking at': 25,
        'on back of': 32,
        'holding': 21,
        'between': 10,
        'laying on': 24,
        'riding': 38,
        'has': 20,
        'across': 2,
        'wearing': 48,
        'walking on': 46,
        'eating': 14,
        'above': 1,
        'part of': 36,
        'walking in': 45,
        'sitting on': 40,
        'under': 43,
        'covered in': 12,
        'carrying': 11,
        'using': 44,
        'along': 4,
        'with': 50,
        'on': 31,
        'covering': 13,
        'of': 30,
        'against': 3,
        'playing': 37,
        'near': 29,
        'painted on': 34,
        'mounted on': 28
    }

    pred_list = [0] * 51
    pred_list[0] = 1
    for k, v in freq_dict.items():
        pred_list[int(pred2idx[k])] = 1

    return pred_list


def get_semantic_diversity():
    freq_dict = {
        "on": 13,
        "has": 12,
        "in": 12,
        "of": 12,
        "wearing": 6,
        "near": 12,
        "with": 12,
        "above": 11,
        "holding": 9,
        "behind": 11,
        "under": 12,
        "sitting on": 9,
        "wears": 4,
        "standing on": 7,
        "in front of": 10,
        "attached to": 9,
        "at": 7,
        "hanging from": 6,
        "over": 10,
        "for": 9,
        "riding": 3,
        "carrying": 5,
        "eating": 3,
        "walking on": 2,
        "playing": 1,
        "covering": 6,
        "laying on": 6,
        "along": 3,
        "watching": 3,
        "and": 10,
        "between": 8,
        "belonging to": 3,
        "painted on": 3,
        "against": 5,
        "looking at": 5,
        "from": 5,
        "parked on": 1,
        "to": 6,
        "made of": 2,
        "covered in": 3,
        "mounted on": 3,
        "says": 1,
        "part of": 5,
        "across": 4,
        "flying in": 1,
        "using": 3,
        "on back of": 5,
        "lying on": 5,
        "growing on": 1,
        "walking in": 2,
    }

    pred2idx = {
        'and': 5,
        'says': 39,
        'belonging to': 9,
        'over': 33,
        'parked on': 35,
        'growing on': 18,
        'standing on': 41,
        'made of': 27,
        'attached to': 7,
        'at': 6,
        'in': 22,
        'hanging from': 19,
        'wears': 49,
        'in front of': 23,
        'from': 17,
        'for': 16,
        'watching': 47,
        'lying on': 26,
        'to': 42,
        'behind': 8,
        'flying in': 15,
        'looking at': 25,
        'on back of': 32,
        'holding': 21,
        'between': 10,
        'laying on': 24,
        'riding': 38,
        'has': 20,
        'across': 2,
        'wearing': 48,
        'walking on': 46,
        'eating': 14,
        'above': 1,
        'part of': 36,
        'walking in': 45,
        'sitting on': 40,
        'under': 43,
        'covered in': 12,
        'carrying': 11,
        'using': 44,
        'along': 4,
        'with': 50,
        'on': 31,
        'covering': 13,
        'of': 30,
        'against': 3,
        'playing': 37,
        'near': 29,
        'painted on': 34,
        'mounted on': 28
    }

    pred_list = [0] * 51
    pred_list[0] = 1
    for k, v in freq_dict.items():
        pred_list[int(pred2idx[k])] = v

    return pred_list


def get_sub_proto_label(mode='concat'):
    semantic_list = get_semantic_diversity()
    idx2concept = range(sum(semantic_list))

    if mode.lower() != "add" and mode.lower() != "concat" and mode.lower() != "clip":
        raise ValueError("Incorrect mode you input it.")
    cluster_dict = torch.load(
        f"/home/xiejunlin/workspace/Intra-Imbalanced-SGG/datasets/datafiles/intra-work/cluster_results/cluster_dict_{mode}.pt")

    return cluster_dict