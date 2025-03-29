import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# importing libraries

import glob
import sys
import gc
from time import time
import cv2


import PIL as pil

import pandas as pd
import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from tqdm import tqdm
import wandb
wandb.login()

sys.path.insert(0,'../src/')

from data_process import *
from cvt import *
from model_train import *

reference_images = pd.read_csv('../input/filename_mapping.csv')

main_catalogue = pd.read_csv('../input/gz2_classes.csv')

model_01_catalogue = pd.DataFrame()
model_01_catalogue['dr7ID'] = main_catalogue['dr7objid']
model_01_catalogue['class'] = main_catalogue['gz2class']

model_01_catalogue.drop(model_01_catalogue[model_01_catalogue['class'] == 'A'].index, inplace=True)

# connecting each class with the corresponding asset_id
model_01_catalogue = model_01_catalogue.merge(
    reference_images[['objid', 'asset_id']], 
    left_on='dr7ID', 
    right_on='objid', 
    how='left'
).drop(columns=['objid'])  # Drop extra 'objid' column after merging
model_01_catalogue = model_01_catalogue.sort_values(by=['asset_id']).reset_index(drop=True)

model_01_catalogue['class'] = model_01_catalogue['class'].apply(lambda x: x.replace('(', '').replace(')', '').ljust(6, '0'))

label_diagram = pd.DataFrame(columns=['r1', 'r2', 'r3', 'r4', 'r5'])
label_diagram['asset_id'] = model_01_catalogue['asset_id']
label_diagram['r1'] = model_01_catalogue['class'].apply(choose_class1)
label_diagram['r2'] = model_01_catalogue['class'].apply(choose_class2)
label_diagram['r3'] = model_01_catalogue['class'].apply(choose_class3)
label_diagram['r4'] = model_01_catalogue['class'].apply(choose_class4)
label_diagram['r5'] = model_01_catalogue['class'].apply(choose_class5)

imgs_path = '../input/images_gz2/images/'
W, H, C = 224, 224, 4

file_list = glob.glob(os.path.join(imgs_path, '*.jpg'))
file_list = sorted(file_list)

# select files whose asset_id is the same as the one in the label_diagram
file_list = [f for f in file_list if int(f.split('/')[-1].split('.')[0]) in label_diagram['asset_id'].values]

file_list = file_list[:30000]

class_mapping = {x : i for i, x in enumerate(sorted(label_diagram['r1'].unique()))}

if __name__ == '__main__':
    gmorph_d = galaxy_img_dataset(file_list, label_diagram, label_mapping='r1', class_mapping=class_mapping)

    total_size = len(gmorph_d)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        gmorph_d, [train_size, val_size, test_size], generator=generator
    )

    epochs = 10
    lr = 0.0001
    tmax = epochs // 3
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    embed_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=16, pin_memory=True
                                )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=16, pin_memory=True
                            )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=16, pin_memory=True
                                )

    gmorph_model = CvT_stride(embed_size, len(class_mapping))
    optimizer = torch.optim.AdamW(gmorph_model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=0.0001)
    loss_func = nn.CrossEntropyLoss()

    results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_loader, val_loader, loss_func, optimizer, scheduler, device, save_name=f'CvT_stride')

    del gmorph_model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()