import torch.nn as nn
from inferno.io.box.posetrack import *
import inferno.io.box.coco as coco
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR
#from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from PoseEstimation.models.callbacks import SaveModelAtBestValidationScore, DumpValidationTags, ToggleGradientTruncation
from collections import OrderedDict

from PoseEstimation.models import posenet
from PoseEstimation.models.loss import *
import PoseEstimation.models.flipped as flipped

from os.path import join, expanduser
from shutil import copyfile

LOG_DIRECTORY = 'runs/sampled_skeleton_regression/011_small-smaller-sigma'
SAVE_DIRECTORY = LOG_DIRECTORY
if os.path.exists(LOG_DIRECTORY):
    assert False, 'log directory exists'
os.mkdir(LOG_DIRECTORY)
copyfile(__file__, join(LOG_DIRECTORY, 'config.txt'))

DATASET_DIRECTORY = join(expanduser('~'), 'Datasets', 'COCO')
USE_CUDA = True
NUM_TARGETS = 5
BATCHSIZE = 12 #12 for small, 6 for big (1080ti) #24 for 4 GPUs

def load_pretrained_model():
    model_config = {
        'nstack': 4,
        'inp_dim': 256,
        'oup_dim': 68,
        'increase': 128
    }
    model = posenet.PoseNet(**model_config)
    state_dict = torch.load('runs/pretrained/checkpoint.pth.tar')['state_dict']
    mapped_dict = OrderedDict()
    for name in state_dict.keys():
        mapped_dict[name[13:]] = state_dict[name]
    model.load_state_dict(mapped_dict)
    model.cuda()
    return model

# Build torch model
tag_dim = 1
model_config = {
    'nstack': 2,
    'inp_dim': 256,  # 128
    'oup_dim': 68 + (1-tag_dim)*17,  # 17+19*4+50 for VV,  # 68,
    'increase': 128,  # 32
    'truncate_grad_flow': False
}
#model = load_pretrained_model()
model = posenet.PoseNet(**model_config)
#model = torch.load('runs/sampled_skeleton_regression/003/model.pytorch')
print('model initialized')

# Initialize trainer
trainer = Trainer(model)

# Built Tensorboard logger
def save_indices(parts):
    result = []
    for part in parts:
        result.append(part)
        result.append(17 + part)
    return result
logger = TensorboardLogger(
    log_scalars_every=(1, 'iterations'),
    log_images_every=(100, 'iterations'),
    send_volume_at_z_indices=save_indices([0, 7, 15])) #[0, 8, 10, 17+4*0, 17+4*9, 17+4*11] #[0, 1, 14, 15, 16, 29])
logger.observe_state('validation_prediction', 'validation')
logger.observe_state('validation_inputs', 'validation')
logger.observe_state('validation_target', 'validation')

# Build loss with logging
weighted_loss = SampledSkeletonRegressionLoss(trainer=trainer, loss_weights=(1, 1, 1, 1))#SkeletonLoss(trainer=trainer)#
weighted_loss.register_logger(logger=logger)
loss = weighted_loss #flipped.FlipLoss(weighted_loss, 1)

# Build trainer
trainer.build_optimizer('Adam', param_groups=None, lr=1e-4)\
    .build_criterion(loss)\
    .validate_every((500, 'iterations'))\
    .register_callback(SaveModelAtBestValidationScore(to_directory=SAVE_DIRECTORY, smoothness=0, verbose=True))\
    .set_max_num_epochs('inf')\
    .build_logger(logger, log_directory=LOG_DIRECTORY)\
    .register_callback(AutoLR(
        factor=0.98,
        patience='100 iterations',
        monitor_while='validating',
        monitor_momentum=.95,
        consider_improvement_with_respect_to='previous')
    )


#    .register_callback(DumpValidationTags(os.path.join(LOG_DIRECTORY, "intermediate_tags.h5"), tag_dim=1))\

# .save_every((1000, 'iterations'), to_directory=SAVE_DIRECTORY)\

# TODO check if tutorial is wrong for saving
# ----------------------> TODO think about weight decay <----------------


# Load loaders
train_loader, validate_loader = coco.get_coco_keypoint_loaders(
    DATASET_DIRECTORY, input_res=512, output_res=128, tag_dim = tag_dim,
    train_batch_size=BATCHSIZE, validate_batch_size=BATCHSIZE, num_workers=8) #n_train=[81]) #15
print('loaders initialized')

# Bind loaders
trainer \
    .bind_loader('train', train_loader, num_inputs=1, num_targets=NUM_TARGETS) \
    .bind_loader('validate', validate_loader, num_inputs=1, num_targets=NUM_TARGETS)


if USE_CUDA:
    trainer.cuda()
    #trainer.cuda(devices=[0,1])

print('beginning training')
# Go!
trainer.fit()
