import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# importing libraries

import glob
import sys
import gc
from time import time
import cv2
from collections import Counter


import PIL as pil

import pandas as pd
import numpy as np
np.random.seed(42)
import random

import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
import wandb
from pytorch_lightning.loggers import WandbLogger
wandb.login()

sys.path.insert(0,'../src/')

import pytorch_lightning as pl

from data_processing import *
from training_loop import *
from labeling_system import *
import cvt as cvt
import cvt_benchmark as cvtb
# ----------------

reference_images = pd.read_csv('../input/filename_mapping.csv')

main_catalogue = pd.read_csv('../input/gz2_classes.csv')

main_catalogue.drop(main_catalogue[main_catalogue['gz2class'] == 'A'].index, inplace=True)

main_catalogue = main_catalogue.merge(
    reference_images[['objid', 'asset_id']], 
    left_on='dr7objid', 
    right_on='objid', 
    how='left'
).drop(columns=['objid'])  # Drop extra 'objid' column after merging
main_catalogue = main_catalogue.sort_values(by=['asset_id']).reset_index(drop=True)
# ===========
print('loaded the catalogues')
soft_label_dict_run1 = create_label_dict1(main_catalogue)
soft_label_dict_run2 = create_label_dict2(main_catalogue)
print('soft labels created')
hard_run1 = create_hard_labels(soft_label_dict_run1)
hard_run2 = create_hard_labels(soft_label_dict_run2)
print('hard labels created')
soft_run1_conf, soft_run1_spur = section_spurious(main_catalogue, soft_label_dict_run1, entropy_threshold=1.5)
print('spurios stuff detected')
runs = [
    {"soft_labels": soft_run1_conf, "hard_labels": hard_run1, "num_classes": len(soft_label_dict_run1[list(soft_label_dict_run1.keys())[0]])},
    {"soft_labels": soft_label_dict_run2, "hard_labels": hard_run2, "num_classes": len(soft_label_dict_run2[list(soft_label_dict_run2.keys())[0]])},
]
# ===========

imgs_path = '../input/images_gz2/images/'
W, H, C = 224, 224, 4

file_list = create_file_list(imgs_path, soft_run1_conf)
print('file list created')
n = 100

img_sub, labels_sub = data_setup(file_list, hard_run1, n)
traino, valido, testo = split_data(img_sub, labels_sub)
print('data setup complete!')
# ==================

# LOOP
# params
epochs = 2
lr = 1e-4
tmax = epochs
device= 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 32
embed_size = 64
use_soft_labels = True
m = 3

wandb_logger = WandbLogger(project="gmorph", name="multiGPU_run")

if __name__ == '__main__':
    print('Loop started')
    
    print('----------------')
    #  INITIAL DATA SETUP
    # =====================

    train_dl, valid_dl, test_dl, y_train, y_valid, y_test = create_data_loaders(traino, valido, testo, hard_run1, bs, aux_train=None, aux_valid=None, aux_test=None, soft_labels_dict=soft_run1_conf)
    
    # SAVING SETUP
    # ======================

    results_runs = []
    results_runs_class = []

    outputs, labels = [], []

    axt = torch.zeros(len(y_train))
    axv = torch.zeros(len(y_valid))
    axte = torch.zeros(len(y_test))

    # ======================
    for i in range(1, m):
        print(f'ITERATION{i}')
        config = runs[i % len(runs)]
        
        
        gmorph_model = cvt.CvT(embed_size, config["num_classes"], hint=(i>1))
        lightning_model = CvTLightning(gmorph_model, lr=lr, use_soft_labels=(i>0))
        
        trainer = pl.Trainer(
            accelerator="gpu",
            devices='auto',  
            strategy="ddp",
            max_epochs=epochs,
            precision=16,
            logger=wandb_logger,
            log_every_n_steps=10
        )

        trainer.fit(lightning_model, train_dl, valid_dl)

        trainer.test(lightning_model, dataloaders=test_dl)

        preds = np.load("../output/test_preds.npy", allow_pickle=True)

        del gmorph_model
        gc.collect()
        torch.cuda.empty_cache()

        # PREP FOR NEXT ROUND
        print('Updating data')
        print('---------------')

        axt += y_train
        axv += y_valid
        axte += torch.tensor(preds)

        train_dl, valid_dl, test_dl, y_train, y_valid, y_test = create_data_loaders(traino, valido, testo, hard_run1, bs, aux_train=None, aux_valid=None, aux_test=None, soft_labels_dict=soft_labels[i])
    
    print('Loop is finished!')

    # =============
    # BENCHMARK
    # =============
    
    benchmark_soft_labels = {
    int(row["asset_id"]): create_benchmark_soft_labels(row)
    for _, row in main_catalogue.iterrows()
    }

    benchmark_hard_labels = {
        asset_id: int(np.argmax(label)) 
        for asset_id, label in benchmark_soft_labels.items()
    }

    train_dl, valid_dl, test_dl = create_data_loaders_bench(traino, valido, testo, benchmark_hard_labels, bs, soft_labels_dict=benchmark_soft_labels)

    benchmark_model = cvtb.CvT_bench(embed_size, num_class=8)
    lightning_benchmark = CvTLightning(benchmark_model, lr=lr, use_soft_labels=False)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        max_epochs=60,
        precision=16,
        logger=wandb_logger,
        log_every_n_steps=10
    )

    trainer.fit(lightning_benchmark, train_dl, valid_dl)
    trainer.test(lightning_benchmark, dataloaders=test_dl)

    del lightning_benchmark
    gc.collect()
    torch.cuda.empty_cache()
    
    print('Completely done!')