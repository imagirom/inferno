import torch
import numpy as np
import importlib
import operator
import itertools

# Helpers when setting up training


def make_input(t, requires_grad=False, need_cuda = True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    if need_cuda:
        inp = inp.cuda()
    return inp


def index_select(tensor, index_tensor):
    shape = tensor.shape
    dim = index_tensor.shape[1]
    factors = torch.LongTensor(list(itertools.accumulate((1,) + shape[1:dim][::-1], operator.mul))[::-1])
    if tensor.is_cuda:
        factors = factors.cuda(tensor.get_device())
    d = 1
    for i in range(dim):
        d *= tensor.shape[i]
    ind = torch.sum(factors*index_tensor, dim=1)
    assert torch.max(ind) < tensor.contiguous().view(d, -1).shape[0]
    return tensor.contiguous().view(d, -1)[ind]


def sample(dist, n_samples=1, return_probabilities=False, normalize_probabilities=True):  # TODO: make device agnostic after torch 0.4.0

    if len(dist.shape) == 1:  # 1D distribution
        cumsum = dist.cumsum(dim=0)
        ind = torch.sum((cumsum[None, :] < cumsum[-1]*torch.rand(n_samples)[:, None]).long(), dim=1)
        if not return_probabilities:
            return ind
        else:
            prob = dist[ind]
            if normalize_probabilities:
                prob = prob / dist.sum()
            return ind, prob

    else:  # multidimensional distribution
        shape = dist.shape
        if not return_probabilities:
            ind = sample(dist.contiguous().view(-1), n_samples, return_probabilities=False)
        else:
            ind, prob = sample(dist.contiguous().view(-1), n_samples, return_probabilities=True,
                               normalize_probabilities=normalize_probabilities)

        result = torch.zeros((n_samples, len(shape))).long()
        for i, s in enumerate(reversed(shape)):
            result[:, -(i+1)] = ind % s
            ind = (ind - result[:, -i]) / s

        if not return_probabilities:
            return result
        else:
            return result, prob


if __name__ == '__main__':
    #t = torch.eye(128, 128)
    #print(sample(t, 20000, True))

    print(10000/3.32e6)
    print(128*128*8000/3.32e6)
    print(128*128)
