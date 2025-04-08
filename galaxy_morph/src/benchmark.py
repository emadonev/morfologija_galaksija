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
import cvt_benchmark as cvtb
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

n = 100
bs = 64
images_orig, labels_orig = data_setup(conf_file_list, hard_run2_conf, n)
traino, valido, testo, y_traino, y_valido, y_testo = split_data(images_orig, labels_orig)

epochs = 1
lr = 2e-5
tmax = epochs
device= 'cuda' if torch.cuda.is_available() else 'cpu'
embed_size = 64

train_dl, valid_dl, test_dl, y_train, y_valid, y_test = create_data_loaders_bench(traino, valido, testo, hard_run2_conf, bs)

gmorph_model = cvtb.CvT_bench(embed_size, 7)
optimizer = torch.optim.NAdam(gmorph_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=2e-6)
loss_func1 = nn.KLDivLoss(reduction='batchmean')
loss_func2 = nn.CrossEntropyLoss()

results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_dl, valid_dl, loss_func1, loss_func2, optimizer, scheduler, device, save_name='benchmark_final')

y_true, preds = test_model(test_dl, gmorph_model, device)

del gmorph_model
del optimizer
gc.collect()
torch.cuda.empty_cache()

outputs = np.array([x.cpu().detach().numpy() for x in preds])
labels = np.array([x.cpu().detach().numpy() for x in y_true])

with open('../output/benchmark/results_runs_bench_final_true.pkl', 'wb') as f:
    pickle.dump(my_dict, results)

with open('../output/benchmark/results_runs_class_bench_final_true.pkl', 'wb') as f:
    pickle.dump(my_dict, results_class)

np.save('../output/benchmark/outputs_test_bench_final_true.npy', outputs, allow_pickle=True)
np.save('../output/benchmark/labels_test_bench_final_true.npy', labels, allow_pickle=True)