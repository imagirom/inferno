from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.essentials import DumpHDF5Every
import h5py as h5
import os
import torch


class SaveModelAtBestValidationScore(Callback):
    """
    Triggers a save at the best EMA (exponential moving average) validation score.
    The basic `Trainer` has built in support for saving at the best validation score, but this
    callback might eventually replace that functionality.
    """
    def __init__(self, to_directory, smoothness=0, verbose=False):
        super(SaveModelAtBestValidationScore, self).__init__()
        # Privates
        self._ema_validation_score = None
        self._best_ema_validation_score = None
        # Publics
        self.smoothness = smoothness
        self.verbose = verbose
        self.to_directory = to_directory

    def end_of_validation_run(self, **_):
        # Get score (i.e. validation error if available, else validation loss)
        current_validation_score = self.trainer.get_state('validation_error_averaged')
        current_validation_score = self.trainer.get_state('validation_loss_averaged') \
            if current_validation_score is None else current_validation_score
        # Maintain ema
        if self._ema_validation_score is None:
            self._ema_validation_score = current_validation_score
            self._best_ema_validation_score = current_validation_score
        else:
            self._ema_validation_score = self.smoothness * self._ema_validation_score + \
                                         (1 - self.smoothness) * current_validation_score
        # This overrides the default behaviour, but reduces to it if smoothness = 0
        self.trainer._is_iteration_with_best_validation_score = \
            self._ema_validation_score <= self._best_ema_validation_score
        # Trigger a save
        if self.trainer._is_iteration_with_best_validation_score:
            if self.verbose:
                self.trainer.print("Current smoothed validation score {} is better "
                                   "than the best smoothed validation score {}."
                                   .format(self._ema_validation_score,
                                           self._best_ema_validation_score))
            self._best_ema_validation_score = self._ema_validation_score
            self.trainer.save_model(self.to_directory)  # This is what I changed
        else:
            if self.verbose:
                self.trainer.print("Current smoothed validation score {} is not better "
                                   "than the best smoothed validation score {}."
                                   .format(self._ema_validation_score,
                                           self._best_ema_validation_score))
        # Done


class DumpValidationPredictions(Callback):
    def __init__(self, filepath):
        super(DumpValidationPredictions, self).__init__()

        self.filepath = filepath
        self.n_val_imgs = 500

        self.current_pos = 0
        self.file_initialized = False

    def initialize_file(self, pred):
        with h5.File(self.filepath, 'w') as f:
            f.create_dataset("validation_outputs", (1, self.n_val_imgs) + tuple(pred.size()[1:]),
                             maxshape = (None, self.n_val_imgs) + tuple(pred.size()[1:]),
                             dtype='f')
        self.file_initialized = True

    def extend_file(self):
        #adjust dset size
        with h5.File(self.filepath, 'r+') as f:
            dset = f['validation_outputs']
            dset.resize(dset.shape[0] + 1, axis=0)

    def begin_of_validation_run(self, **kwargs):
        self.current_pos = 0
        if self.file_initialized:
            self.extend_file()

    def end_of_validation_iteration(self, **kwargs):
        print(kwargs)
        pred = self.trainer.get_state('validation_prediction')
        print('pred shape', len(pred))
        print('pred shape', pred.size())
        if not self.file_initialized:
            self.initialize_file(pred)

        new_pos = self.current_pos + pred.size()[0]
        with h5.File(self.filepath, 'r+') as f:
            dset = f['validation_outputs']
            dset[-1, self.current_pos:new_pos] = pred

        self.current_pos = new_pos


class DumpValidationTags(Callback):
    def __init__(self, filepath, tag_dim):
        super(DumpValidationTags, self).__init__()

        self.filepath = filepath
        self.n_val_imgs = 500
        self.tag_dim = tag_dim

        self.current_pos = 0
        self.file_initialized = False
        self.n_parts = 17

    def initialize_file(self, pred):
        with h5.File(self.filepath, 'w') as f:
            f.create_dataset("validation_tags", (1, self.n_val_imgs) + tuple(pred.size()[1:]),
                             maxshape = (None, self.n_val_imgs) + tuple(pred.size()[1:]),
                             dtype='f')
        self.file_initialized = True

    def extend_file(self):
        #adjust dset size
        with h5.File(self.filepath, 'r+') as f:
            dset = f['validation_tags']
            dset.resize(dset.shape[0] + 1, axis=0)

    def begin_of_validation_run(self, **kwargs):
        self.current_pos = 0
        if self.file_initialized:
            self.extend_file()

    def extract_tags(self, preds, kpts):
        tags = preds[:, :, self.n_parts:(self.tag_dim+1)*self.n_parts]
        preds_flat = tags.contiguous().view(tags.size()[:-3] + (self.tag_dim, -1))
        # shape: batchsize, n_stack, tag_dim, n_joints * n_pixels
        result = torch.zeros((kpts.size()[0], preds.size()[1]) + kpts.size()[1:-1] + (1+self.tag_dim,))
        # shape: max_n_people, batchsize, n_joints, tag_dim + 1

        #print('res', result.shape)
        #print('tags', tags.shape)
        #print('preds_flat', preds_flat.shape)
        for n_img in range(len(kpts)):
            for i_stack in range(preds.size()[1]):
                for i, kpt in enumerate(kpts[n_img]):
                    for j, pos in enumerate(kpt):
                        if pos[-1] > 0:
                            result[n_img, i_stack, i, j, -1] = 1
                            result[n_img, i_stack, i, j, :-1] = preds_flat[n_img, i_stack, :, int(pos[0])]

        return result

    def end_of_validation_iteration(self, **_):
        pred = self.trainer.get_state('validation_prediction')
        target = self.trainer.get_state('validation_target')
        #print('target', target[1].size())
        #print('pred shape', len(pred))
        #print('pred shape', pred.size())
        kpts = target[1]
        tags = self.extract_tags(pred, kpts)

        if not self.file_initialized:
            self.initialize_file(tags)

        new_pos = self.current_pos + pred.size()[0]
        with h5.File(self.filepath, 'r+') as f:
            dset = f['validation_tags']
            dset[-1, self.current_pos:new_pos] = tags

        self.current_pos = new_pos


class ToggleGradientTruncation(Callback):
    def __init__(self, epoch):
        super(ToggleGradientTruncation, self).__init__()

        self.epoch = epoch

    def end_of_validation_run(self, **kwargs):
        if self.trainer.epoch_count == self.epoch:
            self.trainer.model.truncate_grad_flow = False
            print('grad flow no longer truncated')

class ChangeLoss(Callback):
    def __init__(self, epoch, new_loss):
        super(ChangeLoss, self).__init__()
        self.epoch = epoch
        self.new_loss = new_loss

    def end_of_validation_run(self, **kwargs):
        if self.trainer.epoch_count == self.epoch:
            print('changing loss function')
            self.epoch = -1
            self.trainer.build_criterion(self.new_loss)
