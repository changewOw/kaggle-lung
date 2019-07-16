


import numpy as np
import gc
from matplotlib import pyplot as plt
from DataGen import datagenerator
from loss import bce_dice_loss, my_iou_metric
from augmentation import AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST
from segmentation_mo import Xnet
from keras import backend as K
from my_callbacks import SnapshotCallbackBuilder

class Trainer:
    def __init__(self, batch_size, folds, model_name, epochs,steps_per_epoch, img_size=256):
        K.clear_session()
        self.batch_size = batch_size
        self.kfold_train_path = './folds/train_{}.npy'.format(folds)
        self.kfold_valid_path = './folds/valid_{}.npy'.format(folds)
        self.img_size = img_size
        self.epochs = epochs
        self.snapshotcallback = SnapshotCallbackBuilder(folds=folds, steps_per_epoch=steps_per_epoch ,init_lr=0.05)

        self.model = self._create_model(model_name)

        self.train_gen = datagenerator(self.kfold_train_path, self.batch_size,
                                       augmentations=AUGMENTATIONS_TRAIN, img_size=self.img_size)
        self.valid_gen =datagenerator(self.kfold_valid_path, self.batch_size,
                                      augmentations=AUGMENTATIONS_TEST, img_size=img_size)
        print('dd')
    def _create_model(self, model_name):
        if model_name == 'Xnet':
            model = Xnet(backbone_name='resnet34', input_shape=(self.img_size, self.img_size, 3), decoder_block_type='upsampling')
        return model

    def train(self):
        self.model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[my_iou_metric])

        history = self.model.fit_generator(generator=self.train_gen,
                                           validation_data=self.valid_gen,
                                           epochs=self.epochs,
                                           verbose=1,
                                           callbacks=self.snapshotcallback.get_callbacks())
        return history

    def plot(self, history):
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['my_iou_metric'][1:])
        plt.plot(history.history['val_my_iou_metric'][1:])
        plt.ylabel('iou')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')

        plt.title('model IOU')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'][1:])
        plt.plot(history.history['val_loss'][1:])
        plt.ylabel('val_loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.title('model loss')
        gc.collect()

if __name__ == '__main__':
    folds = 0
    batch_size = 4
    dataset_size = len(np.load('./folds/train_{}.npy'.format(folds)))

    steps_per_epoch = int(np.ceil(dataset_size / batch_size))
    del dataset_size
    gc.collect()

    trainer = Trainer(batch_size, folds, 'Xnet', 150, steps_per_epoch=steps_per_epoch)
    history = trainer.train()
    trainer.plot(history)