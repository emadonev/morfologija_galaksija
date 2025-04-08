import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
import gc
from time import time
from collections import Counter
import pandas as pd
import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
import wandb
import pickle

wandb.login()

sys.path.insert(0,'../src/')

# -----
from datap_efficient import *
from model_train_benchmark import *
from labeling_system import *
import cvt_benchmark_attn as cvtb
# -----
main_runs = pd.read_csv('../input/main_runs.csv')
# ------

soft_label_dict_run1 = create_label_dict1(main_runs)
soft_label_dict_run2 = create_label_dict2(main_runs)

soft_run1_conf, soft_run1_spur = section_spurious(soft_label_dict_run1, num=3)
soft_run2_conf, soft_run2_spur = section_spurious(soft_label_dict_run2, num=7)

hard_run1_conf = create_hard_labels(soft_run1_conf)
hard_run2_conf = create_hard_labels(soft_run2_conf)

print(Counter(list(hard_run1_conf.values())))
print(Counter(list(hard_run2_conf.values())))

imgs_path = '../input/images_gz2/images/'
W, H, C = 224, 224, 4

conf_file_list = create_file_list(imgs_path, soft_run1_conf, soft_run2_conf)

n = 1000
bs = 128
embed_size = 64
device= 'cuda' if torch.cuda.is_available() else 'cpu'

images_orig, labels_orig = data_setup(conf_file_list, hard_run2_conf, n)
traino, valido, testo, y_traino, y_valido, y_testo = split_data(images_orig, labels_orig)

train_iter, valid_iter, test_iter = create_dali_iterators(traino, valido, testo, hard_run2_conf, bs)

model_path = '../output/benchmark/model_benchmark_test_maps_epoch7.pth'
model = cvtb.CvT_bench(embed_size, 7)    
state_dict = torch.load(model_path)      
model.load_state_dict(state_dict)         
model.eval() 

print('making attention maps')
cvtb.cvt_attention_map(model, test_iter, device='cuda', dest_dir='../output/benchmark/')