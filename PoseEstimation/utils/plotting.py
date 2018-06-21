import numpy as np
import matplotlib.pyplot as plt


def imgrid(imgs, ax=None, padding=1, pad_value=.5, **kwargs):
    """
    expects 4D array, last two axis being y and x of the image
    """
    if ax is None:
        ax = plt.gca()
    plt.axis('off')
    padded = pad_value * np.ones((imgs.shape[0], imgs.shape[1], imgs.shape[2]+padding, imgs.shape[3]+padding),
                                 dtype=imgs.dtype)
    padded[:, :, :-padding, :-padding] = imgs
    return ax.imshow(np.concatenate(np.concatenate(padded, axis=-1), axis=-2)[:-padding, :-padding], **kwargs)


def magic_colorbar(im):
    plt.colorbar(im, fraction=0.046, pad=0.04)
