import torch
import torch.nn as nn
from torch.autograd import Variable
import inferno.utils.torch_utils as thu
import time


class SkeletonGenerator(nn.Module):
    def __init__(self, sigma=1, softmax_factor=5):
        super(SkeletonGenerator, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.value_softmax = nn.Softmax(dim=1)
        self.softmax_factor = softmax_factor
        self.sigma = sigma

    def generate_masks(self, tag_images, tag_values):
        # shape of tag_images: n_stack, n_parts, tag_dim, dim_x, dim_y
        # shape of tag_values: n_stack, n_values, tag_dim

        n_stack, n_parts, dim_x, dim_y, tag_dim  = tag_images.shape
        n_values, tag_dim = tag_values.shape[1:]
        assert tag_images.shape[-1] == tag_dim

        # generate masks
        tag_images = tag_images.contiguous().view((n_stack , 1       , n_parts, dim_x, dim_y, tag_dim))
        tag_values = tag_values.contiguous().view((n_stack , n_values, 1      , 1    , 1    , tag_dim))

        masks = -torch.sum((tag_images - tag_values)**2, dim=-1) / (2*self.sigma)

        return masks


    def forward(self, heatmaps, tag_images, tag_values, return_masked_heatmaps=False):
        # shape of heatmaps: n_stack, n_parts, dim_x, dim_y
        # shape of tag_images: n_stack, n_parts, dim_x, dim_y, tag_dim
        # shape of tag_values: n_stack, n_values, tag_dim

        n_stack, n_parts, dim_x, dim_y  = heatmaps.shape
        n_values, tag_dim = tag_values.shape[1:]
        assert tag_images.shape[-1] == tag_dim
        #print(n_stack, n_parts, n_values, tag_dim dim_x, dim_y)

        # generate masks
        masks = self.generate_masks(tag_images, tag_values)

        # mask heatmaps, apply softmax
        heatmaps = heatmaps.contiguous().view((n_stack, 1, n_parts, dim_x, dim_y))
        masked_heatmaps = masks + heatmaps
        masked_heatmaps = masked_heatmaps.contiguous().view((-1, dim_x * dim_y))
        # shape is now (n_stack * n_parts * n_values, dim_x * dim_y),
        masked_heatmaps = self.softmax(self.softmax_factor * masked_heatmaps)
        masked_heatmaps = masked_heatmaps.contiguous().view((-1, dim_x, dim_y))

        # apply soft argmax
        x = torch.linspace(0, 1, dim_x).cuda() # TODO: consider subpixel offset. is the first pixel really 0 and the last 1?
        # TODO: consider multiple devices
        y = torch.linspace(0, 1, dim_y).cuda() # TODO: consider subpixel offset. is the first pixel really 0 and the last 1?
        xx = x[:,None].repeat(1, dim_y)
        yy = y[None,:].repeat(dim_x, 1)
        yx = Variable(torch.stack([yy, xx], dim=-1), requires_grad=False)[None]
        skeleton_coordinates = torch.sum(torch.sum(masked_heatmaps[...,None] * yx, dim=-2), dim=-2)
        skeleton_coordinates = skeleton_coordinates.contiguous().view(n_stack, n_values, n_parts, 2)

        #print(f'skeleton_coordinates: {skeleton_coordinates}')

        # TODO: calculate certainties per keypoint, maybe directly from masked heatmaps

        if not return_masked_heatmaps:
            return skeleton_coordinates
        else:
            return skeleton_coordinates, masked_heatmaps.contiguous().view((n_stack, n_values, n_parts, dim_x, dim_y))

