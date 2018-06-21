import torch
from torch import nn

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class FlipNet(nn.Module):
    def __init__(self, net, dim):
        super(FlipNet, self).__init__()
        self.net = net
        self.dim = dim

    def forward(self, input):
        out = self.net(input)
        flipped_input = flip(input, self.dim)
        out_flipped = flip(self.net(flipped_input), self.dim)
        return torch.stack([out, out_flipped])


class FlipLoss(nn.Module):
    def __init__(self, loss, flipFactor):
        super(FlipLoss, self).__init__()
        self.loss = loss
        self.flipFactor = flipFactor
        self.flipLoss = nn.MSELoss()

    def forward(self, preds, labels):
        return self.loss(preds[0], labels) + self.flipFactor * self.flipLoss(preds[0]-preds[1], torch.zeros_like(preds[0]))
