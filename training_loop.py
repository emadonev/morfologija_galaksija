from typing import Any


import os
import glob
import sys
sys.path.insert(0,'../src/')

import gc
from time import time

import wandb

import pandas as pd
import numpy as np

np.random.seed(42)

import matplotlib.pyplot as plt

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from data_processing import *
import cvt as cvt

# ==============

class CvTLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-4, num_classes=4, use_soft_labels=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.KLDivLoss(reduction='batchmean') if use_soft_labels else nn.CrossEntropyLoss()
        self.use_soft_labels = use_soft_labels
        self.num_classes = num_classes

        self.val_preds = []
        self.val_targets = []

        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        if self.use_soft_labels:
            log_probs = F.log_softmax(logits, dim=1)
            loss = self.loss_fn(log_probs, y)
        else:
            loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y.argmax(dim=1)).float().mean() if self.use_soft_labels else (preds == y).float().mean()

        targets = torch.argmax(y, dim=1) if self.use_soft_labels else y

        precision = precision_score(targets.cpu(), preds.cpu(), average='macro', zero_division=0)
        recall = recall_score(targets.cpu(), preds.cpu(), average='macro', zero_division=0)
        f1 = f1_score(targets.cpu(), preds.cpu(), average='macro', zero_division=0)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        
        self.log_dict({
            "train_loss": loss,
            "train_acc": acc,
            "train_precision": float(precision),
            "train_recall": float(recall),
            "train_f1": float(f1),
        }, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Use CrossEntropyLoss which expects targets as indices
        loss = nn.CrossEntropyLoss()(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        precision = precision_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
        recall = recall_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

        precision_c = precision_score(y.cpu().numpy(), preds.cpu().numpy(), average=None, zero_division=0)
        recall_c = recall_score(y.cpu().numpy(), preds.cpu().numpy(), average=None, zero_division=0)
        f1_c = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average=None, zero_division=0)

        self.log_dict({
            "val_loss": loss,
            "val_acc": acc,
            "val_precision": float(precision),
            "val_recall": float(recall),
            "val_f1": float(f1),
        }, prog_bar=True, sync_dist=True)

        for i, p in enumerate(precision_c):
            self.log(f"val_precision_c{i}", float(p), prog_bar=True, sync_dist=True)
        for i, p in enumerate(recall_c):
            self.log(f"val_recall_c{i}", float(p), prog_bar=True, sync_dist=True)
        for i, p in enumerate(f1_c):
            self.log(f"val_f1_c{i}", float(p), prog_bar=True, sync_dist=True)

        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)

        # Define total number of steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)  # 10% of steps for warmup

        def warmup_cosine(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=warmup_cosine),
            "interval": "step",  # step-wise scheduler
            "frequency": 1
        }

        return [optimizer], [scheduler]

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()

        np.save("../output/val_preds_epoch{:02d}.npy".format(self.current_epoch), preds)
        np.save("../output/val_targets_epoch{:02d}.npy".format(self.current_epoch), targets)

        # Optional: clear lists
        self.val_preds = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        preds = logits.argmax(dim=0)
        targets = torch.argmax(y, dim=0) if self.use_soft_labels else y

        acc = (preds == targets).float().mean()

        precision = precision_score(targets.cpu(), preds.cpu(), average='macro', zero_division=0)
        recall = recall_score(targets.cpu(), preds.cpu(), average='macro', zero_division=0)
        f1 = f1_score(targets.cpu(), preds.cpu(), average='macro', zero_division=0)

        self.log_dict({
            "test_acc": acc,
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
        }, prog_bar=True, sync_dist=True)

        self.test_preds.append(preds.cpu())
        self.test_targets.append(targets.cpu())

        return acc

    def on_test_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()

        np.save("../output/test_preds.npy", preds)
        np.save("../output/test_targets.npy", targets)

        self.test_preds = []
        self.test_targets = []