from keras.utils import Sequence
import numpy as np
from cv2 import resize
from PIL import Image

class datagenerator(Sequence):
    def __init__(self, kfold_path, batch_size, augmentations=None,
                 img_size=256, n_channels=3, shuffle=True):
        # kfold_path: i.e. './folds/train_0.npy'
        # like this './folds/valid_0.npy'
        self.batch_size = batch_size
        self.train_path = np.load(kfold_path)
        # self.mask_path = [fn.replace('./train', './masks') for fn in self.train_path]
        self.augment = augmentations
        self.img_size = img_size
        self.shuffle = shuffle
        self.n_channels = n_channels


        self.indexes = np.arange(len(self.train_path))
        self.on_epoch_end()
        # print('dd')
    #
    def __len__(self):
        return int(np.ceil(len(self.train_path) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_img = self.train_path[indexes]
        X, y = self.data_generation(list_IDs_img)

        if self.augment is None:
            return X, np.array(y) / 255
        else:
            img, mask = [], []
            for x, y in zip(X,y):
                augmented = self.augment(image=x,mask=y)
                img.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(img), np.array(mask) / 255


    def data_generation(self, list_IDs_img):
        X = np.empty((len(list_IDs_img), self.img_size, self.img_size, self.n_channels))
        # 注意keras计算loss传进去的是(?,?,?,?)默认4个维度，若要传进三维的
        # 则需要用target_tensor=K.placeholder...
        # 但是计算交叉熵需要与pred相同shape 反正一致即可，怎么都行
        y = np.empty((len(list_IDs_img), self.img_size, self.img_size, 1))

        for i, img_path in enumerate(list_IDs_img):
            mask_path = img_path.replace('./train', './masks')

            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            if len(img.shape) == 2:
                img = np.repeat(img[..., None], 3,2)

            X[i,] = resize(img, (self.img_size, self.img_size))
            y[i,] = resize(mask,(self.img_size, self.img_size))[..., np.newaxis]

            y[y>0] = 255 # resize有可能改变mask的值确保mask只有0和255

        return np.uint8(X), np.uint8(y)






if __name__ == '__main__':
    a = datagenerator('./folds/train_0.npy', 32)
    a.__getitem__(0)