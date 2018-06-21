import torch
import torch.nn as nn
import torch.nn.functional as F
from PoseEstimation.models.layers import Conv, Hourglass


class GroupNet(nn.Module):
    def __init__(self, n_proposals, n_joints, n_split, inp_dim, increase):
        super(GroupNet, self).__init__()
        self.n_joints = n_joints
        self.n_proposals = n_proposals
        if n_split is 1:
            self.pre = nn.Sequential(
                Conv(inp_dim=n_joints, out_dim=inp_dim, kernel_size=3),
            )
        else:
            self.pre = nn.Sequential(
                Conv(inp_dim=n_joints*2*n_split, out_dim=inp_dim, kernel_size=3),
                #Conv(inp_dim=64, out_dim=inp_dim, kernel_size=3)
            )

        self.features = nn.Sequential(
            Hourglass(4, inp_dim, bn=False, increase=increase),
            #Conv(inp_dim, inp_dim, 3, bn=False),
            #Conv(inp_dim, inp_dim, 3, bn=False),
            Hourglass(4, inp_dim, bn=False, increase=increase),
            Conv(inp_dim, inp_dim, 3, bn=False),
            #Conv(inp_dim, inp_dim, 3, bn=False),
        )
        self.post = nn.Sequential(
            #Conv(inp_dim=inp_dim, out_dim=inp_dim, kernel_size=3),
            Conv(inp_dim=inp_dim, out_dim=(n_proposals + 1)*n_joints, kernel_size=3)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.features(x)
        x = self.post(x)
        #y = x[:, :-1]
        #y = x.shape[1] * F.normalize(y)
        #x = torch.cat([y, x[:,-1:]], dim=1)
        return x.contiguous().view((x.shape[0], self.n_proposals + 1, self.n_joints) + x.shape[-2:])


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, truncate_grad_flow=False, **_):
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.features = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
            Conv(inp_dim, inp_dim, 3, bn=False),
            Conv(inp_dim, inp_dim, 3, bn=False)
        ) for i in range(nstack)] )

        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )

        self.nstack = nstack
        self.truncate_grad_flow = True

    def forward(self, imgs):
        #x = imgs.permute(0, 3, 1, 2) # here, the color dimension is moved to the front.
        x = self.pre(imgs)
        preds = []
        for i in range(self.nstack):
            feature = self.features[i](x)
            preds.append( self.outs[i](feature) )
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds[-1]) + self.merge_features[i](feature)
            if self.truncate_grad_flow:
                x = Variable(x.data)

        return torch.stack(preds, 1)
