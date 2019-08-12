import numpy as np
import numbers
import random
import os
from PIL import Image, ImageStat
import keras
from keras.utils import to_categorical

train_size = 448
import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, Resize,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, RandomBrightnessContrast,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop
)

AUGMENTATIONS_TRAIN = Compose([
    Resize(train_size, train_size, p=1),  # 必须执行resize到448
    HorizontalFlip(p=0.5),
    OneOf([
        RandomGamma(),
        RandomBrightnessContrast(),
    ], p=0.2),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion()
    ], p=0.2),
    ShiftScaleRotate(shift_limit=0.1, rotate_limit=15, scale_limit=0.1, p=0.3)
], p=1)
AUGMENTATIONS_WARMUP = Compose([Resize(train_size, train_size, p=1), HorizontalFlip(p=0.5)], p=1)
AUGMENTATIONS_TEST = Compose([Resize(train_size, train_size, p=1)], p=1)


class Cutout(object):
    def __init__(self, n_holes, max_height, max_width, min_height=None, min_width=None,
                 fill_value_mode='zero', p=0.5):
        self.n_holes = n_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_width = min_width if min_width is not None else max_width
        self.min_height = min_height if min_height is not None else max_height
        self.fill_value_mode = fill_value_mode  # 'zero' 'one' 'uniform'
        self.p = p
        assert 0 < self.min_height <= self.max_height
        assert 0 < self.min_width <= self.max_width
        assert 0 < self.n_holes
        assert self.fill_value_mode in ['zero', 'one', 'uniform']

    def __call__(self, img, semantic_label):
        # 不能够破坏图片标签
        mask_pixel_count = semantic_label.sum()  # 计数label总标签数
        if np.random.rand() > self.p:
            return img, semantic_label

        h = img.shape[0]
        w = img.shape[1]

        if self.fill_value_mode == 'zero':
            f = np.zeros
            param = {'shape': (h, w, 3)}
        elif self.fill_value_mode == 'one':
            f = np.one
            param = {'shape': (h, w, 3)}
        else:
            f = np.random.uniform
            param = {'low': 0, 'high': 255, 'size': (h, w, 3)}

        cut_mask = np.ones((h, w), dtype=np.bool)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            h_l = np.random.randint(self.min_height, self.max_height + 1)
            w_l = np.random.randint(self.min_width, self.max_width + 1)

            y1 = np.clip(y - h_l // 2, 0, h)
            y2 = np.clip(y + h_l // 2, 0, h)
            x1 = np.clip(x - w_l // 2, 0, w)
            x2 = np.clip(x + w_l // 2, 0, w)

            if mask_pixel_count > 0:
                cut_mask_pixel_count = semantic_label[y1:y2, x1:x2].sum()
                if cut_mask_pixel_count / mask_pixel_count > 0.5:
                    continue
            cut_mask[y1:y2, x1:x2] = 0

        img = np.where(np.tile(np.expand_dims(cut_mask, axis=-1), (1, 1, 3)), img, f(**param))

        semantic_label = np.where(cut_mask, semantic_label, 0)

        return np.uint8(img), np.uint8(semantic_label)
class DataGenerator(keras.utils.Sequence):
    def __init__(self, mode, folds, augmentations=None, batch_size=32,
                 n_channels=3, shuffle=True, train_size=448,
                 debug=False, cutout=None, mixup=None):
        self.mode = mode
        self.data_filename = np.load('./folds/{}_{}.npy'.format(mode, folds))
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data_filename))

        if self.shuffle:
            np.random.shuffle(self.indexes)

        self._MEAN = np.array([0.5126277, 0.5126277, 0.5126277], dtype=np.float32).reshape(1, 1, 1, 3)
        self._STD = np.array([0.2530869, 0.2530869, 0.2530869], dtype=np.float32).reshape(1, 1, 1, 3)

        self.train_size = train_size
        self.debug = debug
        self.cutout = cutout
        self.mixup = mixup

        print(batch_size)
        print('dd')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_im = self.data_filename[indexes]

        X, y, mask = self.data_generator(list_IDs_im)

        if self.debug:
            return X, y, mask

        return self.precess_input(X), y

    def data_generator(self, list_IDs_im):
        imgs, label, label_adv, swap_law = [], [], [], []
        if self.debug:
            masks = []
        for fn in list_IDs_im:
            img_path = './siimacr-pneumothorax-segmentation-data-512/train/' + fn
            mask_path = './siimacr-pneumothorax-segmentation-data-512/masks/' + fn

            sample = self.data_one(img_path, mask_path)
            if self.mode == 'train':
                imgs.append(sample[0])
                imgs.append(sample[1])
                label.append(sample[2])
                label.append(sample[2])
                label_adv.append(sample[3])
                label_adv.append(sample[4])
                swap_law.append(sample[5])
                swap_law.append(sample[6])
                if self.debug:
                    masks.append(sample[-1])
                    masks.append(sample[-1])
            else:
                imgs.append(sample[0])
                label.append(sample[1])
                label_adv.append(sample[2])
                swap_law.append(sample[3])
                if self.debug:
                    masks.append(sample[-1])
        if self.debug:
            return np.array(imgs), [np.array(label), np.array(label_adv), np.array(swap_law)], np.array(masks)

    def data_one(self, img_path, mask_path):
        # 获取原图和gt_label
        img = np.array(Image.open(img_path).convert('RGB'))  # numpy array
        mask = np.array(Image.open(mask_path))

        mask[mask > 0] = 1

        crop_num = [7, 7]

        if self.mode == 'train':
            img, mask = self.aug_img(img, mask)  # aug必须加上resize到448!!!!!
            gt = 1 if mask.max() > 0 else 0
            img_unswap = Image.fromarray(img)  # PIL image

            image_unswap_list = crop_image(img_unswap, crop_num)
            swap_law1 = [(i - 24) / 49 for i in range(crop_num[0] * crop_num[1])]

            img_swap = swap(img_unswap)
            image_swap_list = crop_image(img_swap, crop_num)
            unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list]
            swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]
            swap_law2 = []
            for swap_im in swap_stats:
                # ?寻找l1最相似的块(最相似的块意味着相同) 贴上标签
                distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
                index = distance.index(min(distance))
                swap_law2.append((index - 24) / 49)
            label = to_categorical([gt], 2)[0]
            label_s = to_categorical([1], 2)[0]  # to_categorical([gt], num_class*2)[0]
            label_swap = to_categorical([0], 2)[0]  # to_categorical([label_swap], num_class*2)[0]

            return [np.array(img_unswap), np.array(img_swap), label, label_s, label_swap, swap_law1, swap_law2, mask]
        else:
            gt = 1 if mask.max() > 0 else 0
            img, mask = self.aug_img(img, mask)  # aug必须加上resize到448!!!!!
            img_unswap = Image.fromarray(img)  # PIL image
            swap_law1 = [(i - 24) / 49 for i in range(crop_num[0] * crop_num[1])]

            label = to_categorical([gt], 2)[0]
            label_s = to_categorical([1], 2)[0]  # to_categorical([gt], num_class * 2)[0]
            return [np.array(img_unswap), label, label_s, swap_law1, mask]

    def aug_img(self, img, mask):
        augmented = self.augmentations(image=img, mask=mask)
        return augmented['image'], augmented['mask']

    def precess_input(self, img_b):
        img_b = img_b / 255.0
        img_b -= self._MEAN
        img_b /= self._STD
        return img_b


def crop_image(image, cropnum):
    width, high = image.size
    crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
    crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
    im_list = []
    for j in range(len(crop_y) - 1):
        for i in range(len(crop_x) - 1):
            im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
    return im_list


def swap(img):
    img = randomSwap(img, (7, 7))
    return img


def randomSwap(img, size):
    """
    :param img:  PIL 类型
    :param size:
    :return:
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
        size = size

    widthcut, highcut = img.size
    img = img.crop((10, 10, widthcut - 10, highcut - 10))
    images = crop_image(img, size)
    pro = 5
    if pro >= 5:
        tmpx = []
        tmpy = []
        count_x = 0
        count_y = 0
        k = 1
        RAN = 2
        for i in range(size[1] * size[0]):
            tmpx.append(images[i])
            count_x += 1
            if len(tmpx) >= k:
                tmp = tmpx[count_x - RAN:count_x]
                random.shuffle(tmp)
                tmpx[count_x - RAN:count_x] = tmp
            if count_x == size[0]:
                tmpy.append(tmpx)
                count_x = 0
                count_y += 1
                tmpx = []
            if len(tmpy) >= k:
                tmp2 = tmpy[count_y - RAN:count_y]
                random.shuffle(tmp2)
                tmpy[count_y - RAN:count_y] = tmp2
        random_im = []
        for line in tmpy:
            random_im.extend(line)

        # random.shuffle(images)
        width, high = img.size
        iw = int(width / size[0])
        ih = int(high / size[1])
        toImage = Image.new('RGB', (iw * size[0], ih * size[1]))
        x = 0
        y = 0
        for i in random_im:
            i = i.resize((iw, ih), Image.ANTIALIAS)
            toImage.paste(i, (x * iw, y * ih))
            x += 1
            if x == size[0]:
                x = 0
                y += 1
    else:
        toImage = img
    toImage = toImage.resize((widthcut, highcut))
    return toImage



if __name__ == '__main__':
    a = DataGenerator('valid', 0, batch_size=64, shuffle=False, debug=True, augmentations=AUGMENTATIONS_TEST)
    # a = DataGenerator('train', 0, batch_size=32, augmentations=AUGMENTATIONS_TRAIN, shuffle=False, debug=True)
    images, label, masks = a.__getitem__(0)
    binary_label, label_adv_debug, swap_law_debug = label
    print(images.shape) # (64,448,448,3)
    print(masks.shape) # (64,448,448)
    print(binary_label.shape) # (64,2)
    print(label_adv_debug.shape) # (64,2)
    print(swap_law_debug.shape) # (64, 49)