
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from PIL import Image


all_mask_fn = glob.glob('./masks/*')
mask_df = pd.DataFrame()
mask_df['file_names'] = all_mask_fn
mask_df['mask_percentage'] = 0
mask_df.set_index('file_names', inplace=True)
for fn in all_mask_fn:
    mask_df.loc[fn,'mask_percentage'] = np.array(Image.open(fn)).sum()/(256*256*255)
mask_df.reset_index(inplace=True)
# sns.distplot(mask_df.mask_percentage)
mask_df['labels'] = 0
mask_df.loc[mask_df.mask_percentage>0,'labels'] = 1

kfolder = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)

all_train_fn = np.array([fn.replace('./masks','./train') for fn in mask_df['file_names'].values])

for i, (k_train_idx, k_valid_idx) in enumerate(kfolder.split(all_train_fn, mask_df.labels.values)):
    print(len(k_train_idx), len(k_valid_idx))
    # print("there is {} equal to 1".format(mask_df['labels'][k_labels].sum()))
    # print("there is {} equal to 0".format(len(k_labels) - mask_df['labels'][k_labels].sum()))
    np.save('./folds/train_{}.npy'.format(i), all_train_fn[k_train_idx])
    np.save('./folds/valid_{}.npy'.format(i), all_train_fn[k_valid_idx])





