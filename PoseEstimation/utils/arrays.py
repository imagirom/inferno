import numpy as np


def circle_mask(shape, position, radius):
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    yx = np.flip(np.stack(np.meshgrid(x, y), axis=-1), axis=-1)
    result = (np.sum((yx-position)**2, axis=2) < radius**2).astype(np.float32)
    return result
