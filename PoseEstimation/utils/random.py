import numpy as np
import torch

def arg_sample(dist):
    normed_dist = dist/np.max(dist)
    while True:
        x = np.random.randint(0, len(dist))
        y = np.random.rand(1)
        if y < normed_dist[x]:
            break
    return x


def ranf_scaled(shape=1, lower=0, upper=1):
    return lower + (upper - lower) * np.random.ranf(shape)
