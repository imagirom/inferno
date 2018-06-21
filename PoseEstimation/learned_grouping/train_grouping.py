import torch.nn as nn
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR
from PoseEstimation.models.callbacks import SaveModelAtBestValidationScore
from collections import OrderedDict
from os.path import join, expanduser
import os
from shutil import copyfile

from PoseEstimation.learned_grouping.grouping_loss import GroupingLoss
from PoseEstimation.learned_grouping.toy_dataset import *
from PoseEstimation.learned_grouping.grouping_model import GroupNet


LOG_DIRECTORY = '../runs/toy_grouping/matching/11'
USE_CUDA = True
NUM_TARGETS = 2
BATCHSIZE = 5
LEARNING_RATE = 1e-5
NTRAIN = 10000
NVAL = 500
loss_weights = (25.0, 0.0, 0.0, 1.0)

# Build torch model
model_config = {
    'inp_dim': 128,#128,
    'increase': 128,  # 32
    'n_joints': 2,
    'n_proposals': 16,
    'n_split': 1
}
LOG_DIRECTORY = LOG_DIRECTORY + f'_lr{LEARNING_RATE}' + f'_weights{loss_weights}'
SAVE_DIRECTORY = LOG_DIRECTORY
if os.path.exists(LOG_DIRECTORY):
    assert False, 'log directory exists'
os.mkdir(LOG_DIRECTORY)
copyfile(__file__, join(LOG_DIRECTORY, 'config.txt'))

model = GroupNet(**model_config)
print('model initialized')



# Initialize trainer
trainer = Trainer(model)


logger = TensorboardLogger(log_scalars_every=(1, 'iterations'),
                           log_images_every=(100, 'iterations'))
logger.observe_state('training_loss', 'train')
logger.observe_state('validation_inputs', 'validation')
logger.observe_state('validation_prediction', 'validation')
logger.observe_state('validation_inputs', 'validation')
logger.observe_state('validation_target', 'validation')

# Build loss with logging
loss_config = {
    'output_mode': 'bg_separate',
    'push_mode': 'matching_onehot',
    'pull_mode': 'squared_diff'
}

weighted_loss = GroupingLoss(trainer=trainer, choice_dict=loss_config, loss_weights=loss_weights)  # SkeletonLoss(trainer=trainer)#
weighted_loss.register_logger(logger=logger)
loss = weighted_loss  # flipped.FlipLoss(weighted_loss, 1)

# Build trainer
trainer.build_optimizer('Adam', param_groups=model.parameters(), lr=LEARNING_RATE) \
    .build_criterion(loss) \
    .validate_every((1000, 'iterations')) \
    .register_callback(SaveModelAtBestValidationScore(to_directory=SAVE_DIRECTORY, smoothness=0, verbose=True)) \
    .set_max_num_epochs('inf') \
    .build_logger(logger, log_directory=LOG_DIRECTORY) \
    .register_callback(AutoLR(factor=0.98,
                              patience='100 iterations',
                              monitor_while='validating',
                              monitor_momentum=.95,
                              consider_improvement_with_respect_to='previous')
                       )

#    .register_callback(DumpValidationTags(os.path.join(LOG_DIRECTORY, "intermediate_tags.h5"), tag_dim=1))\

# .save_every((1000, 'iterations'), to_directory=SAVE_DIRECTORY)\

# Load loaders
train_loader, val_loader = get_toy_loaders(n_joints=model_config['n_joints'], n_train=NTRAIN, n_val=NVAL, batch_size=BATCHSIZE,
                                           max_n_persons=2, min_n_persons=2, n_split=model_config['n_split'])
print('loaders initialized')

# Bind loaders
trainer \
    .bind_loader('train', train_loader, num_inputs=1, num_targets=NUM_TARGETS) \
    .bind_loader('validate', val_loader, num_inputs=1, num_targets=NUM_TARGETS)

if USE_CUDA:
    trainer.cuda()
    # trainer.cuda(devices=[0,1])

print('beginning training')
# Go!
trainer.fit()
