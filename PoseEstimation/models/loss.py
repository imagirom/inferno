import torch
from torch.autograd import Variable
import torch.nn.functional as F
import inferno.utils.torch_utils as thu
import time
import numpy as np
from ..utils.torch_utils import make_input, sample, index_select

from ..extensions.AE.AE_loss import AEloss
from .skeleton_generation import SkeletonGenerator


class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    mask is used to mask off the crowds in coco dataset
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt, masks):
        assert pred.size() == gt.size()
        l = ((pred - gt)**2) * masks[:, None, :, :].expand_as(pred)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l

class VectorLoss(torch.nn.Module):
    """
    loss for voting vectors
    """
    def __init__(self):
        super(VectorLoss, self).__init__()

    def forward(self, pred, gt, masks):
        pred = pred.contiguous().view_as(gt)
        l = ((pred - gt)**2) * masks[:, :, None, None, :, :].expand_as(pred)
        l = l.mean(dim=5).mean(dim=4).mean(dim=3).mean(dim=2).mean(dim=1) #TODO think about sum/mean
        return l

def singleTagLoss(pred_tag, keypoints, relative_to_position=False):
    """
    associative embedding loss for one image
    """
    n_joints = keypoints.shape[1]
    # output resolution
    res = int(np.sqrt(pred_tag.shape[1]/n_joints))
    eps = 1e-6
    tags = []
    pull = 0
    for i in keypoints:  # iterate over persons in image
        tmp = []
        for j in i:
            if j[1]>0:
                if not relative_to_position:
                    tmp.append(pred_tag[:, j[0]])
                else:
                    pixel = j[0] % (res*res)
                    y = (pixel // res) / res
                    x = (pixel % res) / res
                    scale = 10 # TODO: find good value
                    offset = scale * Variable(torch.FloatTensor(np.array((x, y) + (0,)*(pred_tag.shape[0]-2)))).cuda(pred_tag.get_device())
                    tag_to_append = pred_tag[:, j[0]] + offset
                    tmp.append(tag_to_append)
        if len(tmp) == 0:
            continue
        tmp = torch.stack(tmp)
        tags.append(tmp.mean(0))
        pull = pull + torch.mean(torch.pow((tmp - tags[-1].expand_as(tmp)), 2))

    if len(tags) == 0:
        return make_input(torch.zeros([1]).float()), make_input(torch.zeros([1]).float())

    tags = torch.stack(tags)#[:, 0]

    num = tags.size()[0]
    #print('tags:', tags.size())
    size = (num, num, tags.size()[1])
    A = tags.unsqueeze(dim=1).expand(*size)
    #print('A', A.shape)
    B = A.permute(1, 0, 2)

    diff = A - B
    #print('diff 1', diff.shape)
    diff = torch.pow(diff, 2).sum(dim=2)#[:, :, 0]
    #print('diff 2', diff.shape)

    push = torch.exp(-diff)  # TODO: why exp and not eg. (1 - abs)**2

    #diff = torch.sqrt(torch.abs(diff))
    max_push = 1
    #print(diff)
    #push = torch.max(torch.stack([torch.abs(max_push - diff), torch.zeros_like(diff)], 0), 0)[0]
    #print(push)
    push = (torch.sum(push) - max_push * num)  # account for diagonal
    #print(push)
    #push = torch.abs(1-diff, 2)

    return push/((num - 1) * num + eps) * 0.5, pull/(num + eps)


def tagLoss(tags, keypoints, relative_to_position=False):
    """
    accumulate the tag loss for each image in the batch
    """
    pushes, pulls = [], []
    keypoints = keypoints.cpu().data.numpy()
    for i in range(tags.size()[0]):
        push, pull = singleTagLoss(tags[i], keypoints[i%len(keypoints)], relative_to_position)
        pushes.append(push)
        pulls.append(pull)
    return torch.stack(pushes)[:, 0], torch.stack(pulls)[:, 0]


def singleSupervisedTagLoss(pred_tag_imgs, keypoints, gt_tags):
    """
    for one image
    """
    #print('kpst', keypoints.shape)
    pred_tags = pred_tag_imgs.contiguous().view(-1)[keypoints[..., 0].contiguous()
        .view(-1).type(torch.cuda.LongTensor)]\
        .contiguous().view_as(keypoints[..., 0])

    #print('pred', pred_tags.shape)
    #print('gt', gt_tags.shape)
    #print('kpts[...,1]', keypoints[..., 1].type_as(pred_tags).size())
    l = ((pred_tags - gt_tags) ** 2) * keypoints[..., 1].type_as(pred_tags)  # TODO check if factor is correct (and not 2)
    l = l.sum(dim=0).sum(dim=0)
    return l


def supervisedTagLoss(pred_tag_imgs, keypoints, gt_tags):
    """
    accumulate the tag loss for each image in the batch
    """
    losses = []
    # print('pred tag images size', pred_tag_imgs.size())
    for i in range(pred_tag_imgs.size()[0]):
        l = []
        for dim in range(len(gt_tags[i])):
            l.append(singleSupervisedTagLoss(pred_tag_imgs[i, dim], keypoints[i], gt_tags[i, dim]))
        losses.append(torch.stack(l, dim=0).mean(dim=0))
    #print('batch losses', losses[0].size(), len(losses))
    return torch.stack(losses)


def loss_list_free_tags(preds, labels, relative_to_position=False):
    assert len(labels) >= 4

    #myAEloss = AEloss()
    heatmapLoss = HeatmapLoss()

    masks = labels[0]
    keypoints = labels[-2]
    #print('kpts', keypoints.shape)
    heatmaps = labels[-3]

    if len(labels) == 4:
        tag_dim = 1
    else:
        tag_dim = int(labels[-1].data.cpu().numpy()[0])
    batchsize = preds.size()[0]
    nstack = preds.size()[1]
    n_parts = heatmaps.size()[1]
    dets = preds[:, :, :n_parts]
    tags = preds[:, :, n_parts:(1+tag_dim)*n_parts].contiguous().view(batchsize, nstack, tag_dim, -1)
    keypoints = keypoints.cpu().long()

    tag_loss = []
    for i in range(nstack):
        tag = tags[:, i]#.contiguous().view(batchsize, -1, tag_dim, 1)
        #tag_loss.append(myAEloss(tag, keypoints))

        #print('tag', tag.shape)
        push, pull = tagLoss(tag, keypoints, relative_to_position)
        tag_loss.append(torch.stack([push, pull], dim=1))
        #print('diff: ', torch.stack([push, pull], dim=1) - tag_loss[-1].cuda())
    tag_loss = torch.stack(tag_loss, dim=1).cuda(tags.get_device())

    detection_loss = []
    for i in range(nstack):
        detection_loss.append( heatmapLoss(dets[:, i], heatmaps, masks) )
    detection_loss = torch.stack(detection_loss, dim=1)

    return [tag_loss[:, :, 0], tag_loss[:, :, 1], detection_loss]  # each is a list with length nstack


def loss_list_supervised_tags(preds, labels):
    assert len(labels) >= 4

    heatmapLoss = HeatmapLoss()

    masks = labels[0]
    keypoints = labels[1]
    heatmaps = labels[3]
    gt_tags = labels[2]

    #gt_tags shape: (batchsize, tag_dim, max_num_people, n_bodyparts)
    tag_dim = gt_tags.size()[1]


    nstack = preds.size()[1]

    n_parts = heatmaps.size()[1]
    dets = preds[:, :, :n_parts]
    batchsize = len(dets)
    tags = preds[:, :, n_parts:(1+tag_dim)*n_parts].contiguous().view(batchsize, nstack, tag_dim, -1)

    #keypoints = keypoints.cpu().long()

    tag_loss = []
    for i in range(nstack):
        tag_loss.append(supervisedTagLoss(tags[:, i], keypoints, gt_tags))
    tag_loss = torch.stack(tag_loss, dim=1)#.cuda(tags.get_device())

    detection_loss = []
    for i in range(nstack):
        detection_loss.append( heatmapLoss(dets[:, i], heatmaps, masks) )
    detection_loss = torch.stack(detection_loss, dim=1)
    #print('tag_loss ', tag_loss.size())
    return [tag_loss, detection_loss]  # each is a list with length nstack


def loss_list_vector_voters(preds, labels):
    assert len(labels) >= 5

    heatmapLoss = HeatmapLoss()
    vectorLoss = VectorLoss()

    masks = labels[0]
    keypoints = labels[1]
    heatmaps = labels[2]
    batchsize = len(heatmaps)
    gt_vec = labels[3]#.view((batchsize,-1,)+heatmaps.size()[2:])
    vec_masks = labels[4]

    nstack = preds.shape[1]

    n_parts = heatmaps.shape[1]
    dets = preds[:, :, :n_parts]
    vec = preds[:, :, n_parts:n_parts+19*2*2] #TODO remove hard coded dim

    vec_loss = []
    for i in range(nstack):
        vec_loss.append(vectorLoss(vec[:, i], gt_vec, vec_masks))
    vec_loss = torch.stack(vec_loss, dim=1)

    detection_loss = []
    for i in range(nstack):
        detection_loss.append( heatmapLoss(dets[:, i], heatmaps, masks) )
    detection_loss = torch.stack(detection_loss, dim=1)
    #print('tag_loss ', tag_loss.size())
    return [vec_loss, detection_loss]  # each is a list with length nstack


def get_average_tags(pred_tag, keypoints):
    """
    get the average tags per person
    """
    n_joints = keypoints.shape[1]
    tags = []
    for i in keypoints:  # iterate over persons in image
        tmp = []
        for j in i:
            if j[1]>0:
                tmp.append(pred_tag[j[0], :])
        if len(tmp) == 0:
            continue
        tmp = torch.stack(tmp)
        tags.append(tmp.mean(0))

    if len(tags) == 0:
        #print('no tags in image')
        return make_input(torch.zeros((1,pred_tag.shape[-1])).float()) #TODO think more carfully about what to do here

    return torch.stack(tags)


def singleSkeletonLoss(heatmaps, tag_imgs, gt_keypoints, gt_skeletons):
    tag_dim = tag_imgs.shape[-1]

    person_mean_tags = []
    for i_stack in range(heatmaps.shape[0]):
        person_mean_tags.append(get_average_tags(tag_imgs[i_stack].contiguous().view(-1, tag_dim), gt_keypoints))
    person_mean_tags = torch.stack(person_mean_tags, dim=0).cuda(device=tag_imgs.get_device())

    skeleton_generator = SkeletonGenerator(sigma=.5, softmax_factor=30)  # TODO: tinker with hyperparams
    skeletons = skeleton_generator(heatmaps, tag_imgs, person_mean_tags)

    gt_skeletons = gt_skeletons[:skeletons.shape[1]]
    diff = torch.sum((gt_skeletons[None, :, :, :2] - skeletons)**2, dim=-1)
    #torch.sqrt(torch.sum((gt_skeletons[None, :, :, :2] - skeletons)**2, dim=-1)) # L1 loss TODO: think about loss carefully: L1/L2, mean/sum..
    ind = (gt_skeletons[None, ..., 2] > 0)
    #diff = diff * (ind.float()) not needed
    print(torch.mean(diff[ind]))
    return torch.mean(diff[ind])


def loss_list_skeleton_comparison(preds, labels):
    assert len(labels) >= 4

    gt_masks = labels[0]
    gt_keypoints = labels[-2]
    gt_skeletons = labels[1]
    #print('kpts', keypoints.shape)
    gt_heatmaps = labels[-3]

    if len(labels) == 4:
        tag_dim = 1
    else:
        tag_dim = int(labels[-1].data.cpu().numpy()[0])

    n_parts = gt_heatmaps.shape[1]
    pred_heatmaps = preds[:, :, :n_parts]
    batchsize, n_stack, n_parts, dim_x, dim_y  = pred_heatmaps.shape
    pred_tag_imgs = preds[:, :, n_parts:(1+tag_dim)*n_parts].contiguous().view(batchsize, n_stack, n_parts, dim_x, dim_y, tag_dim) # THIS IS BAD. TAG_DIM CANNOT BE LAST
    gt_keypoints = gt_keypoints.cpu().long().data

    skeleton_loss = []
    for i in range(batchsize):
        skeleton_loss.append(singleSkeletonLoss(pred_heatmaps[i], pred_tag_imgs[i], gt_keypoints[i], gt_skeletons[i]))

    return [torch.stack(skeleton_loss, dim=1)]


def print_var_info(x, label='var'):
    print(f"{label}: shape={x.shape}, type={type(x)}")


def loss_list_sampled_regression(preds, labels):
    assert len(labels) >= 4

    gt_masks = labels[0]
    gt_keypoints = labels[-2]
    gt_skeletons = labels[1]
    gt_heatmaps = labels[-3]

    if len(labels) == 4:
        tag_dim = 1
    else:
        tag_dim = int(labels[-1].data.cpu().numpy()[0])

    n_parts = gt_heatmaps.shape[1]
    pred_heatmaps = preds[:, :, :n_parts]
    batchsize, n_stack, n_parts, dim_x, dim_y = pred_heatmaps.shape
    pred_tag_imgs = preds[:, :, n_parts:(1 + tag_dim) * n_parts].contiguous()\
        .view(batchsize, n_stack, tag_dim, n_parts, dim_x, dim_y,)

    pred_tags = pred_tag_imgs.contiguous().view(batchsize, n_stack, tag_dim, -1)

    # ------------------------ detection loss ------------------------
    prob_pred_heatmaps = F.sigmoid(pred_heatmaps)
    detection_loss = []
    for i in range(n_stack):
        detection_loss.append(F.binary_cross_entropy(
            input=prob_pred_heatmaps[:, i] * gt_masks[:, None],
            target=(gt_heatmaps > np.e ** -1).float(),
            weight=10 * (gt_heatmaps > np.e ** -1).float() + 5e-1 * (gt_heatmaps < np.e ** -1).float()
        ))
    detection_loss = torch.stack(detection_loss, dim=1)

    #det_loss = torch.nn.NLLLoss(weight=Variable(torch.FloatTensor([1,1e-3])).cuda())
    #for i in range(n_stack):
    #    inp = torch.stack([prob_pred_heatmaps[:, i] * gt_masks[:, None], 1-prob_pred_heatmaps[:, i] * gt_masks[:, None]], dim=1)
    #    target = (gt_heatmaps > np.e ** -1).float(),
    #    detection_loss.append(det_loss(
    #        input=inp,
    #        target=
    #        weight=10 * (gt_heatmaps > np.e ** -2).float() + 1e-1 * (gt_heatmaps < np.e ** -3).float()
    #    ))
    #

    # --------------------------- tag loss ---------------------------
    tag_loss = []
    gt_keypoints = gt_keypoints.cpu().long()
    for i in range(n_stack):
        tag = pred_tags[:, i]
        push, pull = tagLoss(tag, gt_keypoints, False)
        tag_loss.append(torch.stack([push, pull], dim=1))
    tag_loss = torch.stack(tag_loss, dim=1).cuda(pred_tag_imgs.get_device())

    # ------------------- skeleton regression loss -------------------
    n_samples = 10
    ind = torch.stack(
        [torch.stack(
            [sample(prob_pred_heatmaps[j, i].data.cpu(), n_samples) for i in range(n_stack)], dim=0
        ) for j in range(batchsize)], dim=0
    ).cuda()
    # shape of ind: (batchsize, n_stack, 3)

    # decide on corresponding persons
    max_dist = dim_x + dim_y
    person_ids = np.empty((batchsize, n_stack, n_samples), dtype=np.int32)
    for i in range(batchsize):
        skeletons = gt_skeletons[i]
        for j in range(n_stack):
            for k in range(n_samples):
                n_part = ind[i, j, k, 0]
                pos = ind[i, j, k, 1:].float()
                _, person_id = torch.min(torch.norm(max_dist * (skeletons[:, n_part, 2] == 1)[:, None].float() +
                                         skeletons[:, n_part, 1:] - Variable(pos[None]), dim=1), dim=0)
                person_ids[i, j, k] = person_id.data.cpu().numpy()

    # extract tags at positions
    batch_id = torch.arange(0, batchsize).cuda()[:,  None, None].expand(
        (batchsize, n_stack, 10)).contiguous().view(-1, 1).long()
    stack_id = torch.arange(0, n_stack).cuda()[None, :, None].expand(
        (batchsize, n_stack, 10)).contiguous().view(-1, 1).long()
    ind = torch.cat([batch_id, stack_id, ind.view(-1, 3)], dim=1)
    tag_values = index_select(pred_tag_imgs.permute(0, 1, 3, 4, 5, 2), ind)\
        .contiguous().view((batchsize, n_stack, n_samples, tag_dim))

    # generate predicted skeletons
    skeleton_generator = SkeletonGenerator(sigma=.2, softmax_factor=3)
    skeletons = skeleton_generator(
        heatmaps=pred_heatmaps.contiguous().view((batchsize * n_stack, n_parts, dim_x, dim_y)),
        tag_images=pred_tag_imgs.permute(0, 1, 3, 4, 5, 2).contiguous().view((batchsize * n_stack, n_parts, dim_x, dim_y, tag_dim)),
        tag_values=tag_values.contiguous().view((batchsize * n_stack, n_samples, tag_dim))
    ).contiguous().view(batchsize, n_stack, n_samples, n_parts, 2)

    # compute skeleton comparison loss
    def huber_loss(x, delta=.1):
        a = torch.abs(x)
        i = (a < delta).float()
        return i * x**2 + (1-i) * delta * (a - delta/2)
    diffs = Variable(torch.zeros((batchsize, n_stack, n_samples)).float().cuda())
    for i in range(batchsize):
        for j in range(n_stack):
            for k in range(n_samples):
                diff = huber_loss(torch.norm(gt_skeletons[i, person_ids[i, j, k], :, :2] -
                                   skeletons[i, j, k], dim=-1) * (gt_skeletons[i, person_ids[i, j, k], :, 2] > 0).float()
                                   , delta=.1).mean()
                diffs[i, j, k] = diff

    skeleton_regression_loss = diffs.mean(dim=2)
    return [tag_loss[:, :, 0], tag_loss[:, :, 1], detection_loss, skeleton_regression_loss]


class WeightedLoss(torch.nn.Module):

    def __init__(self, loss_list, loss_weights, trainer=None, loss_names=None):
        super(WeightedLoss, self).__init__()
        self.loss_list = loss_list
        self.loss_weights = loss_weights
        self.trainer = trainer
        if loss_names is None:
            loss_names = [str(i) for i in range(len(loss_weights))]
        self.loss_names = loss_names
        self.n_losses = len(loss_weights)
        self.logging_enabled = False

    def forward(self, preds, labels):
        losses = self.loss_list(preds, labels)
        loss = 0
        for i, current in enumerate(losses):
            loss = loss + self.loss_weights[i] * torch.mean(current)
        self.save_losses(losses)
        return loss

    def save_losses(self, losses):
        if (self.trainer is None) or (self.logging_enabled is False):
            return
        for i, current in enumerate(losses):
            self.trainer.update_state(self.get_loss_name(i), thu.unwrap(torch.mean(current)))

    def register_logger(self, logger):  # logger should be a tensorboard logger
        for i in range(self.n_losses):
            logger.observe_state(self.get_loss_name(i, training=True), 'training')
            logger.observe_state(self.get_loss_name(i, training=False), 'validation')
        self.logging_enabled = True

    def get_loss_name(self, i, training=None):
        if training is None:
            assert self.trainer is not None
            assert self.trainer.model_is_defined
            training = self.trainer.model.training
        if training:
            return 'training_' + self.loss_names[i]
        else:
            return 'validation_' + self.loss_names[i]

    def __getstate__(self):  # TODO make this nicer
        """Return state values to be pickled."""
        if self.trainer is None:
            return self
        return WeightedLoss(self.loss_list, self.loss_weights, trainer=None, loss_names=self.loss_names)


class LossFreeTags(WeightedLoss):

    LOSS_NAMES = ['push_loss', 'pull_loss', 'heatmap_loss']  # TODO:check if order is correct

    def __init__(self, relative_to_position=False, loss_weights=(1e-3, 1e-3, 1), **super_kwargs):
        super(LossFreeTags, self).__init__(
            loss_list=lambda preds, lables: loss_list_free_tags(preds, lables, relative_to_position),
            loss_weights=loss_weights,
            loss_names=self.LOSS_NAMES,
            **super_kwargs)


class LossSupervisedTags(WeightedLoss):

    LOSS_NAMES = ['tag_loss', 'heatmap_loss']

    def __init__(self, loss_weights=(1e-3, 1), **super_kwargs):
        super(LossSupervisedTags, self).__init__(
            loss_list=loss_list_supervised_tags,
            loss_weights=loss_weights,
            loss_names=self.LOSS_NAMES,
            **super_kwargs)


class LossVectorVoters(WeightedLoss):

    LOSS_NAMES = ['vector_loss', 'heatmap_loss']

    def __init__(self, loss_weights=(40, 1), **super_kwargs):
        super(LossVectorVoters, self).__init__(
            loss_list=loss_list_vector_voters,
            loss_weights=loss_weights,
            loss_names=self.LOSS_NAMES,
            **super_kwargs)


class SkeletonLoss(WeightedLoss):

    LOSS_NAMES = ['skeleton_loss']

    def __init__(self, loss_weights=(1,), **super_kwargs):
        super(SkeletonLoss, self).__init__(
            loss_list=loss_list_skeleton_comparison,
            loss_weights=loss_weights,
            loss_names=self.LOSS_NAMES,
            **super_kwargs)


class SampledSkeletonRegressionLoss(WeightedLoss):

    LOSS_NAMES = ['push_loss', 'pull_loss', 'heatmap_loss', 'skeleton_loss']

    def __init__(self, loss_weights=(1, 1, 1, 10), **super_kwargs):
        super(SampledSkeletonRegressionLoss, self).__init__(
            loss_list=loss_list_sampled_regression,
            loss_weights=loss_weights,
            loss_names=self.LOSS_NAMES,
            **super_kwargs)
