import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
from PoseEstimation.utils.plotting import imgrid
from PoseEstimation.utils.arrays import circle_mask
from PoseEstimation.utils.random import ranf_scaled
from inferno.io.transform.keypoints import KeypointsNormalization


class ToyDataset(data.Dataset):
    def __init__(self, n_samples, n_joints=3, min_n_persons=2, max_n_persons=2, img_shape=(32, 32), joint_transform=None, stochastic=True):
        super(data.Dataset, self).__init__()
        self.n_samples = n_samples
        self.img_shape = img_shape
        self.max_n_persons = max_n_persons
        self.min_n_persons = min_n_persons
        self.n_joints = n_joints
        self.pad = 3
        self.kpt_radius = 3
        self.joint_transform = joint_transform
        self.stochastic = stochastic
        self.data = self.generate_data(self.n_samples)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx, stochastic=None):
        if stochastic is None:
            stochastic = self.stochastic
        if not stochastic:
            return self.preprocess(self.data[idx])
        else:
            heatmaps, bgs, kpts = self.data[idx]
            kpts_new = kpts + np.concatenate([
                np.random.randint(-self.kpt_radius + 1, self.kpt_radius + 1 - 1, kpts.shape[:-1]+(2,)),
                np.zeros(kpts.shape[:-1]+(1,))
            ], axis=-1)
            return self.preprocess(heatmaps, bgs, kpts_new)

    def preprocess(self, heatmaps, bgs, kpts):
        if self.joint_transform is not None:
            return self.joint_transform(heatmaps, bgs, kpts)
        else:
            return heatmaps, bgs, kpts

    def generate_data(self, n_samples):
        np.random.seed()
        n_joints = self.n_joints

        def one_sample():
            kpt_radius = self.kpt_radius
            n_persons = np.random.randint(self.min_n_persons, self.max_n_persons + 1)
            person_tags = ranf_scaled(n_persons, lower=.5, upper=1)
            x = ranf_scaled((n_persons, n_joints), self.pad, self.img_shape[0] - self.pad)
            y = ranf_scaled((n_persons, n_joints), self.pad, self.img_shape[1] - self.pad)
            kpts = np.round(np.stack([x, y], axis=2))

            # make sure circles do not overlap
            for i, kpt in enumerate(kpts):
                if i == 0:
                    continue
                for joint in range(n_joints):
                    min_dist = np.min(np.linalg.norm(kpts[:i, joint] - kpts[i, joint], axis=-1))
                    while min_dist <= 2 * kpt_radius + 1:
                        new_kpt = np.round(np.array([
                            ranf_scaled(1, self.pad, self.img_shape[0]-self.pad),
                            ranf_scaled(1, self.pad, self.img_shape[1]-self.pad)
                        ]).flatten())
                        kpts[i, joint] = new_kpt
                        min_dist = np.min(np.linalg.norm(kpts[:i, joint] - kpts[i, joint], axis=-1))

            per_person_heatmaps = np.empty((n_persons, n_joints,) + self.img_shape)
            for k, person in enumerate(kpts):
                for i, kpt in enumerate(person):
                    per_person_heatmaps[k, i] = circle_mask(self.img_shape, kpt, kpt_radius)
            heatmaps = np.max(per_person_heatmaps * person_tags[:, None, None, None], axis=0)

            bgs = 1 - np.max(per_person_heatmaps, axis=0)

            kpts = np.concatenate([kpts, np.ones_like(kpts[..., 0])[..., None]], axis=-1) # add visibility for consitency with COCO
            return heatmaps.astype(np.float32), bgs.astype(np.float32), kpts.astype(np.float32)

        return [one_sample() for i in range(n_samples)]

    def show(self, idx=None, sample=None):
        if sample is None:
            assert idx is not None
            sample = self[idx]
        heatmaps, bgs, kpts = sample
        plt.figure(figsize=(12, 4))
        imgrid(heatmaps[:, None], vmin=0, vmax=1)
        plt.figure(figsize=(12, 4))
        imgrid(bgs[:, None], vmin=0, vmax=1)
        plt.show()


def partition_of_one(n, length, sharpness=1):
    x = np.linspace(0, 1, length)[:,None].repeat(n, axis=1)
    x = x + np.linspace(0, -1, n)[None]
    x = np.exp(-(sharpness*n*x)**2)
    x = x/np.linalg.norm(x, axis=1)[:,None]
    return x


class ToyDatasetSplitCoords(ToyDataset):
    def __init__(self, n_split, *args, **kwargs):
        super(ToyDatasetSplitCoords, self).__init__(*args, **kwargs)
        self.n_split = n_split

    def __getitem__(self, idx, *args):
        heatmaps, bgs, kpts = super(ToyDatasetSplitCoords, self).__getitem__(idx, *args)
        res = heatmaps.shape[1]
        extra = partition_of_one(self.n_split, res, sharpness=1)[None].repeat(res, axis=0)
        extra = np.moveaxis(extra, -1, 0)
        extra = np.concatenate([extra, np.moveaxis(extra, 1, 2)], axis=0)
        # heatmaps = np.concatenate([(heatmaps[:, None] * extra [None, :]).reshape(-1, res, res), heatmaps])
        heatmaps = (heatmaps[:, None] * extra[None, :]).reshape(-1, res, res)

        return heatmaps, bgs, kpts


def get_toy_loaders(n_split=1, n_joints=3, min_n_persons=2, max_n_persons=2, n_train=10000, n_val=500, batch_size=5, num_workers=2):
    joint_transforms = KeypointsNormalization(max_num_people=max_n_persons, apply_to=[2], scale_factor=1)

    if n_split==1:
        train_dset = ToyDataset(n_samples=n_train, n_joints=n_joints, max_n_persons=max_n_persons,
                                min_n_persons=min_n_persons, joint_transform=joint_transforms)
        val_dset = ToyDataset(n_samples=n_val, n_joints=n_joints, max_n_persons=max_n_persons,
                              min_n_persons=min_n_persons, joint_transform=joint_transforms)
    else:
        train_dset = ToyDatasetSplitCoords(n_split=n_split, n_samples=n_train, n_joints=n_joints,  min_n_persons=min_n_persons,
                                           max_n_persons=max_n_persons, joint_transform=joint_transforms)
        val_dset = ToyDatasetSplitCoords(n_split=n_split, n_samples=n_val, n_joints=n_joints,  min_n_persons=min_n_persons,
                                         max_n_persons=max_n_persons, joint_transform=joint_transforms)

    train_loader = data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   pin_memory=True)
    validate_loader = data.DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                      pin_memory=True)

    return train_loader, validate_loader


if __name__ == '__main__':
    dset = ToyDatasetSplitCoords(n_split=3, n_samples=100)
    dset.show(0)
