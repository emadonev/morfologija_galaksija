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
from datap_efficient_bench import *
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

n = 5000
bs = 128
images_orig, labels_orig = data_setup(conf_file_list, hard_run2_conf, n)
traino, valido, testo, y_traino, y_valido, y_testo = split_data(images_orig, labels_orig)

epochs = 50
lr = 1e-4
tmax = epochs
device= 'cuda' if torch.cuda.is_available() else 'cpu'
embed_size = 64

train_iter, valid_iter, test_iter = create_dali_iterators(traino, valido, testo, hard_run2_conf, bs)

gmorph_model = cvtb.CvT_bench(embed_size, 7)
optimizer = torch.optim.AdamW(gmorph_model.parameters(), lr=lr, weight_decay=0.04, betas=(0.9, 0.999), eps=1e-8)
warmup_epochs = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    epochs=epochs,
    steps_per_epoch=len(train_iter),
    pct_start=warmup_epochs/epochs,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)

max_grad_norm = 1.0

loss_func2 = nn.CrossEntropyLoss()

results, results_class, train_pred, train_true, train_probs, train_galaxy_ids, valid_pred, valid_true, valid_probs, valid_galaxy_ids = train_model(epochs, gmorph_model, train_iter, valid_iter, loss_func2, optimizer, scheduler, device, max_grad_norm, save_name='benchmark_test_maps')

y_true, y_preds, galaxy_ids = test_model(test_iter, gmorph_model, device)

cvtb.cvt_attention_map(gmorph_model, test_iter, device, dest_dir='../output/benchmark/', sel_gal_ids=None)

del gmorph_model
del optimizer
gc.collect()
torch.cuda.empty_cache()

outputs = np.array([x.cpu().detach().numpy() for x in y_preds])
labels = np.array([x.cpu().detach().numpy() for x in y_true])

with open('../output/benchmark/results_runs_bench_final_true.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('../output/benchmark/results_runs_class_bench_final_true.pkl', 'wb') as f:
    pickle.dump(results_class, f)

np.save('../output/benchmark/outputs_test_bench_final_true.npy', outputs, allow_pickle=True)
np.save('../output/benchmark/labels_test_bench_final_true.npy', labels, allow_pickle=True)