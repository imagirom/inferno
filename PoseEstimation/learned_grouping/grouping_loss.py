import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from PoseEstimation.models.loss import WeightedLoss
from PoseEstimation.utils.torch_utils import index_select


class HLoss(nn.Module):
    def __init__(self, dim=1):
        super(HLoss, self).__init__()
        self.dim = dim

    def forward(self, x):
        b = F.softmax(x, dim=self.dim) * F.log_softmax(x, dim=self.dim)
        b = -b.sum(dim=self.dim).mean()
        return b


class OneHotLoss(nn.Module):
    def __init__(self, eps=10e-12, size_average=True):
        super(OneHotLoss, self).__init__()
        self.eps = eps
        self.size_average = size_average

    def forward(self, a):
        result = torch.sum(torch.prod(1-a, dim=-1)[:,None]/torch.clamp(1-a, max=None, min=self.eps)*a)
        if self.size_average:
            result = result/len(a)
        return result


class SingleGroupingLosses(nn.Module):
    def __init__(self, choice_dict=dict()):
        super(SingleGroupingLosses, self).__init__()
        self.hloss = HLoss(dim=0)
        # weights = torch.zeros(n_proposals + 1).cuda()
        # weights[-1] = 1
        self.bgloss = nn.BCELoss()  # torch.nn.CrossEntropyLoss(weight=weights)#nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.choices = choice_dict
        self.onehot = OneHotLoss(size_average=False)

    def forward(self, heatmaps, bgs, kpts):
        # kpts = torch.LongTensor(kpts)[:, :, :2].cuda()
        n_proposals, n_joints = heatmaps[:-1].shape[:2]
        kpts = kpts.data.long()
        # kpts = kpts[kpts[..., 0, -1] != 0][..., :2] TODO: use this after upgrading pytorch
        # n_persons = len(kpts)
        n_persons = 0
        while True:
            if n_persons == len(kpts) or kpts[n_persons, 0, -1] == 0:
                break
            n_persons += 1
        kpts = kpts[:n_persons, :, :2]
        # entropy loss
        # detection loss
        if 'output_mode' in self.choices:
            output_mode = self.choices['output_mode']
        else:
            output_mode = 'bg_separate'

        if output_mode == 'softmax_all':
            entropy_loss = self.hloss(heatmaps)
            prob_heatmaps = F.softmax(heatmaps, dim=0)  # F.softmax(heatmaps[:-1], dim=0)
            assert not ((prob_heatmaps != prob_heatmaps).any())
            assert prob_heatmaps[-1].shape == bgs.shape, f'{prob_heatmaps[-1].shape} != {bgs.shape}'
        elif output_mode == 'bg_separate':
            entropy_loss = self.hloss(heatmaps[:-1])
            bgs_pred = self.sigmoid(heatmaps[-1])
            prob_heatmaps = torch.cat([F.softmax(heatmaps[:-1], dim=0), bgs_pred[None]], dim=0)

        detection_loss = self.bgloss(prob_heatmaps[-1], bgs)

        # detection_loss = self.bgloss(heatmaps[-1], bgs)

        # matching loss
        kpts = kpts.contiguous().view((-1, 2)).repeat(n_proposals, 1)
        person_ids = torch.arange(0, n_persons).cuda()[:, None].expand((n_persons, n_joints)).contiguous().view(-1,
                                                                                                                1).long()
        proposal_ids = torch.arange(0, n_proposals).cuda()[:, None, None].expand(
            (n_proposals, n_persons, n_joints)).contiguous().view(-1, 1).long()
        part_ids = torch.arange(0, n_joints).cuda()[None, None, :].expand(
            (n_proposals, n_persons, n_joints)).contiguous().view(-1, 1).long()
        ind = torch.cat([proposal_ids, part_ids, kpts], dim=1)

        values = index_select(prob_heatmaps[:-1], ind).contiguous().view(n_proposals, -1)
        diff_mat = values[:, :, None] - values[:, None, :]
        sum_mat = values[:, :, None] + values[:, None, :]
        prod_mat = values[:, :, None] * values[:, None, :]

        same_person_mat = Variable((((person_ids - person_ids.t()) == 0).float()), requires_grad=False)
        different_person_mat = Variable((((person_ids - person_ids.t()) != 0).float()), requires_grad=False)

        # 'push' loss
        if 'push_mode' in self.choices:
            push_mode = self.choices['push_mode']
        else:
            push_mode = 'scalar_product'

        if push_mode == 'difference':
            diff_factor = (1 - torch.exp(-sum_mat)) * different_person_mat
            different_loss = -(1 - (1 - torch.abs(diff_mat)) ** 2)
            different_loss *= diff_factor
            # per_proposal_different_loss = ..
            different_loss = (n_joints ** 2 * n_persons * (n_persons - 1) * (1 - 1 / np.e)) + different_loss.sum() / 2
        elif push_mode == 'scalar_product':
            different_loss = (prod_mat * different_person_mat).sum()
        elif push_mode == 'sum_of_products':
            values = values.contiguous().view(n_proposals, -1, n_joints)
            values = values.permute((0,2,1))
            values = values.contiguous().view(n_proposals * n_joints, -1)
            different_loss = -self.onehot(values)
        elif push_mode == 'matching_onehot':
            det_goodness = values.contiguous().view(n_proposals, -1, n_joints)
            det_goodness = det_goodness.sum(dim=-1) / n_joints
            similarity_matrix = det_goodness**2  # * (1 + det_badness) / 2
            different_loss = -self.onehot(similarity_matrix)

        # 'pull' loss
        if 'pull_mode' in self.choices:
            pull_mode = self.choices['pull_mode']
        else:
            pull_mode = 'squared_diff'

        if pull_mode == 'squared_diff':
            same_loss = torch.sum((diff_mat * same_person_mat) ** 2)
        elif pull_mode == 'abs':
            same_loss = torch.sum(torch.abs(diff_mat * same_person_mat))
        elif pull_mode == 'scalar_product':
            same_loss = same_person_mat[0].sum() - (prod_mat * same_person_mat).sum()

        # print('det', detection_loss)
        # print('ent', entropy_loss)
        # print('same', same_loss)
        # print('diff', different_loss)
        return [detection_loss, entropy_loss, same_loss, different_loss]


class BatchGroupingLosses(nn.Module):  # only batch_size==1 working for now
    def __init__(self, choice_dict={}):
        super(BatchGroupingLosses, self).__init__()
        self.single_loss = SingleGroupingLosses(choice_dict=choice_dict)

    def forward(self, preds, labels):
        #heatmaps = preds.contiguous().view((-1, self.n_joints) + preds.shape[-2:]) TODO: move to model
        heatmaps = preds
        bgs, kpts = labels
        batchsize = len(preds)
        losses = []
        for i in range(batchsize):
            losses.append(self.single_loss(heatmaps[i], bgs[i], kpts[i]))
        return [torch.stack([losses[i][j] for i in range(batchsize)], dim=1) for j in range(4)]


class GroupingLoss(WeightedLoss):

    LOSS_NAMES = ['bg_loss', 'entropy_loss', 'same_loss', 'different_loss']

    def __init__(self, loss_weights=(1.0, 0.0, 1.0, 0.4), choice_dict={}, **super_kwargs):
        loss_list = BatchGroupingLosses(choice_dict=choice_dict)
        super(GroupingLoss, self).__init__(
            loss_list=loss_list,
            loss_weights=loss_weights,
            loss_names=self.LOSS_NAMES,
            **super_kwargs)
