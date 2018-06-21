import numpy as np
#import inferno.io.box.coco as coco
import torch
from torch import nn



def local_greedy_inference(joint_types, confidences, regression_values, confidence_threshold=.3,
                           n_joints=None, joint_ordering=None, person_partitions=None,
                           dist=(lambda dx, confindences: dx*dx)):

    assert len(joint_types) == len(confidences) == len(regression_values)

    if person_partitions is None:
        person_partitions = [np.arange(len(joint_types))]
    print(person_partitions)


    if joint_ordering is None:
        joint_ordering = list[range(n_joints)]
    elif n_joints is None:
        n_joints = len(joint_ordering)
    else:
        assert False

    result = []
    for partition in person_partitions:
        c = confidences[partition]
        f = regression_values[partition]
        types = joint_types[partition]
        assigned_joints = 0
        not_explored = np.ones(len(c), dtype=np.bool)
        while any(not_explored) > 0:
            person = None
            for (joint_number, joint) in enumerate(joint_ordering):
                ind_joint = (types == joint) * not_explored
                c_joint = c[ind_joint]
                f_joint = f[ind_joint]
                ids = partition[ind_joint]
                if len(ids) is 0:
                    continue
                if person is None:
                    i = int(np.argmax(c_joint)),
                    person = np.zeros((n_joints, 2), np.int32)
                else:
                    i = int(np.argmin(dist(f_joint - centroid, c_joint)))
                if c_joint[i] > confidence_threshold:
                    person[joint_number] = [ids[i], True]
                    if assigned_joints == 0:
                        centroid = f_joint[i]
                    else:
                        centroid = (centroid * assigned_joints + f_joint[i]) / (assigned_joints + 1)
                assigned_joints += 1
                not_explored[ids[i]] = False
            result.append(person)
    return np.stack(result)


def parse_heatmap(heatmap, tag_img, , detection_val=0.03, max_detections = 30):
    maxm = nn.MaxPool2d(3, 1, 1).pool(heatmap)
    maxm = torch.eq(maxm, heatmap).float()

def coco_inference(heatmaps, tags, confidence_threshold=.7):
    pass


if __name__ == '__main__':

    n_joints = 3
    joint_ordering = [1, 2, 3]
    joint_types = np.array([1, 1, 1, 2, 2, 3, 3, 3])
    confidences = np.array([1, 0, 1, 1, 1, 1, 1, 1])
    regression_values = np.array([1, 2, 3, 1, 2, 1, 2, 3])
    print(local_greedy_inference(joint_types, confidences, regression_values, joint_ordering=joint_ordering))
