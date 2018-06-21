import torch.nn as nn
from inferno.io.box.posetrack import *
import inferno.io.box.coco as coco
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from PoseEstimation.models.callbacks import SaveModelAtBestValidationScore

from PoseEstimation.models import posenet
from PoseEstimation.models.loss import *

from os.path import join, expanduser


LOG_DIRECTORY = 'pose_track_debug'
SAVE_DIRECTORY = LOG_DIRECTORY
#DATASET_DIRECTORY = join(expanduser('~'), 'Datasets', 'COCO')
DATASET_DIRECTORY = '/export/home/rremme/Datasets/PoseTrack/posetrack_data'
USE_CUDA = True
NUM_TARGETS = 4

# Build torch model
model_config = {
    'nstack': 2,
    'inp_dim': 128,
    'oup_dim': 68,
    'increase': 32
}
model = posenet.PoseNet(**model_config)
print('model initialized')

# Initialize trainer
trainer = Trainer(model)

# Built Tensorboard logger
logger = TensorboardLogger(log_scalars_every=(1, 'iterations'),
                           log_images_every=(100, 'iterations'),
                           send_volume_at_z_indices='all')
logger.observe_state('validation_prediction', 'validation')
logger.observe_state('validation_inputs', 'validation')
logger.observe_state('validation_target', 'validation')

# Build loss with logging
loss = LossSupervisedTags(trainer=trainer)
loss.register_logger(logger=logger)

# Build trainer
trainer.build_optimizer('Adam', param_groups=None, lr=1e-4)\
    .build_criterion(loss)\
    .validate_every((100, 'iterations'))\
    .register_callback(SaveModelAtBestValidationScore(to_directory=SAVE_DIRECTORY, smoothness=0.95, verbose=True))\
    .set_max_num_epochs('inf')\
    .build_logger(logger, log_directory=LOG_DIRECTORY)\
    .register_callback(AutoLR(factor=0.95,
                              patience='100 iterations',
                              monitor_while='validating',
                              monitor_momentum=.95,
                              consider_improvement_with_respect_to='previous')
                       )

# TODO check if tutorial is wrong for saving
# TODO think about weight decay


# Load loaders
train_loader, validate_loader = get_posetrack_loaders(DATASET_DIRECTORY, input_res=512, output_res=128,
                                                    train_batch_size=10, validate_batch_size=1, num_workers=2) #15
print('loaders initialized')

# Bind loaders
trainer \
    .bind_loader('train', train_loader, num_inputs=1, num_targets=NUM_TARGETS) \
    .bind_loader('validate', validate_loader, num_inputs=1, num_targets=NUM_TARGETS)


if USE_CUDA:
    trainer.cuda()

print('beginning training')
# Go!
trainer.fit()
