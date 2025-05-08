import math

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

from models.head.build import build_head
from models.layers.build import build_retrieval_augmentation, build_randomized_smoothing

import yacs.config
from models.backbone.embed_models import BackboneModel


class Model(pl.LightningModule):

    def __init__(self, config: yacs.config.CfgNode, optimizer_config: yacs.config.CfgNode, dataset_ref=None, use_embeddings=True, device="cuda") -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = config['head']['output_dim']

        self.config = config
        self.optimizer_config = optimizer_config
        self.use_embeddings = use_embeddings

        self.input_modality = config['input_modality']
        if isinstance(self.input_modality, list):
            self.stack = True
            config['head']['input_dim'] = 2*config['head']['input_dim']
        elif isinstance(self.input_modality, str):
            self.stack = False
        else:
            raise Exception('Modality format not supported: only list and str allowed')
        # self.output_type = config['output_type']

        self.model = nn.Sequential()

        if "randomized_smoothing" in self.config:
            randomized_smoothing = build_randomized_smoothing(self.config['randomized_smoothing'])
            self.model.add_module('randomized_smoothing', randomized_smoothing)

        if not self.use_embeddings:
            backbone_name = config['backbone']['name']
            backbone = BackboneModel.fromname(backbone_name, device=device).to(device)

            # Freeze backbone
            self.model.add_module('backbone', backbone.eval())


        if "retrieval_augmentation" in config and "name" in config['retrieval_augmentation']:
            if self.stack:
                raise Exception("Stacked input is not supported for retrieval augementation modules")

            retrieval_augmentation = build_retrieval_augmentation(config['retrieval_augmentation'], dataset_ref)
            self.model.add_module('retrieval_augmentation', retrieval_augmentation)

        head = build_head(config['head'])
        self.model.add_module('head', head)

        head_module = list(self.model.children())[-1]
        # Freeze all parameters except those in the head
        for module in self.model.children():
            if module is not head_module:
                for param in module.parameters():
                    param.requires_grad = False

        self.train_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes, average='macro')
        self.val_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes, average='micro')
        self.val_accuracy_macro = Accuracy(task='multiclass', num_classes=self.num_classes, average='macro')

        self.precision = Precision(task='multiclass', num_classes=self.num_classes, average='macro')
        self.recall = Recall(task='multiclass', num_classes=self.num_classes, average='macro')
        self.f1score = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')

    def training_step(self, batch, batch_index):
        (x, y), _ = batch

        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)

        self.train_accuracy(pred, y)
        self.log_dict({
            'train_loss': loss,
            'train_acc': self.train_accuracy
        }, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_index):
        (x, y), _ = batch

        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)

        self.val_accuracy(pred, y)
        self.val_accuracy_macro(pred, y)

        self.precision(pred, y)
        self.recall(pred, y)
        self.f1score(pred, y)

        self.log_dict({
            'val_loss': loss,
            'val_acc': self.val_accuracy,
            'val_acc_macro': self.val_accuracy_macro,
            'val_precision': self.precision,
            'val_recall': self.recall,
            'val_f1score': self.f1score 
        }, sync_dist=True, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model[-1].parameters(), lr=self.optimizer_config['lr'], weight_decay=self.optimizer_config['weight_decay'])
        scheduler = CosineWarmupScheduler(optimizer, self.optimizer_config['warmup'], self.optimizer_config['epochs'])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            }
        }
    
    # def on_save_checkpoint(self, checkpoint):
    #     trainable_state_dict = {k: v for k, v in self.state_dict().items() if v.requires_grad}
    #     checkpoint['state_dict'] = trainable_state_dict    
        
    def forward(self, x):
        # x, _ = self.prep_input_output(x)
        return self.model(x)

class CosineWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

