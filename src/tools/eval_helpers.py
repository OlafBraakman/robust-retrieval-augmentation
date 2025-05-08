import numpy as np
import torch
import torchvision
from math import ceil

from models.meta_arch.model import Model
from models.backbone.embed_models import BackboneModel

from data.datasets.build import build_dataset
from data.preprocessing import get_preprocessor

from functools import partial
from pathlib import Path
from yacs.config import CfgNode

import wandb


class ModelSelector():

    def __init__(self, project="retrieval-augmentation", defense="retrieval_augmentation", type="robust_augmentation"):
        api = wandb.Api()
        raw_runs = api.runs(project)
        self.defense = defense
        self.type = type
        self.runs = np.array([(run.name, run.config, run) for run in raw_runs if run.state == "finished" and run.config['config'][defense] is not None and run.config['config'][defense]['name'] == type])

    def run_filter(self, x, **kwargs):
        try:
            config = x[1]['config']
            
            defense_config = config.get(self.defense, {})
            metadata_args = x[2].metadata['args']
            try:
                seed_index = metadata_args.index("--seed") + 1
                defense_config['seed'] = int(metadata_args[seed_index])
            except:
                print("Seedless run found")

            for key, value in kwargs.items():
                if key in defense_config:
                    if kwargs[key] is None:
                        continue
                    if defense_config[key] != value:
                        return False
                else:
                    return False
            return True
        except:
            return False
    
    def find_model(self, **kwargs):
        return list(filter(partial(self.run_filter, **kwargs), self.runs))

    def load_model(self, config, **kwargs):

        print(kwargs)
        runs = self.find_model(**kwargs)
        
        if not len(runs) == 1:
            raise Exception(f"Found {len(runs)} runs")

        name, model_config, _ = runs[0]

        model_config = CfgNode(model_config['config'])
        # Cast to float to prevent yacs complaining
        model_config['retrieval_augmentation']['alpha'] = float(model_config['retrieval_augmentation']['alpha'])

        config['model']['retrieval_augmentation'].set_new_allowed(True)
        config['model'].merge_from_other_cfg(model_config)

        config['dataset']['split'] = 'train'
        train_dataset_base = build_dataset(config['dataset'])

        config['dataset']['split'] = 'test'
        val_dataset_base = build_dataset(config['dataset'])

        # config['model'] = model_config['config']
        val_dataset_base.preprocessor = get_preprocessor(
            val_dataset_base,
            height=config['model']['backbone']['height'],
            width=config['model']['backbone']['width'],
            phase="test",
        )

        model_path = list(Path("/data/model_checkpoints/").glob(f"{name}*"))[0]
        model = Model.load_from_checkpoint(model_path, dataset_ref=train_dataset_base, config=config['model'], use_embeddings=False, strict=False)
        model.eval()
        
        return model, val_dataset_base
    

class TotalModel(torch.nn.Module):

    def __init__(self, backbone, model, num_classes, device):
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        self.smooth = False
        
        self.model = torch.nn.Sequential()

        if backbone is not None:
            backbone = BackboneModel.fromname(backbone, device=self.device)
            self.model.add_module("backbone", backbone)
            self.model.add_module("classifier", model)
        else:
            self.model.add_module("backbone", model.model.backbone)
            self.model.add_module("classifier", model.model.head)

    def forward(self, x):
        # x = self.preprocessor(x)
        # if not self.smooth:
        return self.model(x)
        # else:
            # return self.smooth_predict_batch(x)

    def computer_loss_pred_batch(self, X, y, loss_fn):
        all_loss = []
        all_pred = []

        assert len(X.shape) == 4

        loss_fn = torch.nn.CrossEntropyLoss()

        for i in range(len(X)):
            loss, pred = self.compute_loss_pred(X[i], y[i], loss_fn)
            all_loss.append(loss)
            all_pred.append(pred)

        return torch.stack(all_loss), torch.stack(all_pred)
                
    def compute_loss_pred(self, x, label, loss_fn, total_samples=1000, mini_batch_size=32, sigma=0.5):
        total_loss = 0.0
        logits = torch.zeros(self.num_classes, device='cpu')

        num_steps = ceil(total_samples / mini_batch_size)
        for step in range(num_steps):
            current_batch_size = min(mini_batch_size, total_samples - step * mini_batch_size)

            batch = x.repeat((current_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch) * sigma

            # Forward pass on the noisy input:
            outputs = self.model(batch + noise)
            logits += outputs.mean(dim=0).cpu()

            # Compute the loss for this mini-batch:
            loss_batch = loss_fn(outputs, label)
            
            # Scale the loss by the number of mini-batches before accumulating
            total_loss += loss_batch / num_steps

        return total_loss, logits.argmax(dim=1)
    
    
    # def _sample_noise(self, x, target, num: int, mini_batch_size, loss_fn, sigma=0.5):

    #     total_loss = 0.0

    #     num_steps = ceil(num / mini_batch_size)
    #     for _ in range(num_steps):
    #         current_batch_size = min(mini_batch_size, num)
    #         num -= current_batch_size
    #         batch = x.repeat((current_batch_size, 1, 1, 1))
    #         noise = torch.randn_like(batch) * sigma

    #         with torch.autocast(device_type="cuda"):  # use mixed precision for forward pass
    #             logits_batch = self.model(batch + noise).mean(dim=0)
    #             loss_batch = loss_fn(logits_batch, target)
    #             total_loss += loss_batch / current_batch_size

    #     return total_loss
            # logits_list.append(logits_batch)
            # torch.cuda.empty_cache()
        # logits = torch.stack(logits_list).mean(dim=0)
        # return logits    

    # def _sample_noise(self, x, num: int, batch_size, sigma=0.5):
    #     # with torch.no_grad():
    #     logits = torch.zeros(self.num_classes, device='cpu')
    #     for _ in range(ceil(num / batch_size)):
    #         this_batch_size = min(batch_size, num)
    #         num -= this_batch_size

    #         batch = x.repeat((this_batch_size, 1, 1, 1))
    #         noise = torch.randn_like(batch) * sigma
    #         logits += self.model(batch + noise).mean(dim=0).cpu()
    #     return logits
    
    # def smooth_loss_batch(self, x, y):
    #     assert len(x.shape) == 4
    #     preds = []

    #     for xi in x:
    #         loss = self._sample_noise(xi, 20, 4)
    #         preds.append(pred)

    #     return torch.stack(preds)

    # def _sample_noise(self, x, num: int, batch_size, sigma=0.5) -> np.ndarray:
    #     """ Sample the base classifier's prediction under noisy corruptions of the input x.

    #     :param x: the input [channel x width x height]
    #     :param num: number of samples to collect
    #     :param batch_size:
    #     :return: an ndarray[int] of length num_classes containing the per-class counts
    #     """
    #     with torch.no_grad():
    #         counts = np.zeros(self.num_classes, dtype=int)
    #         for _ in range(ceil(num / batch_size)):
    #             this_batch_size = min(batch_size, num)
    #             num -= this_batch_size

    #             batch = x.repeat((this_batch_size, 1, 1, 1))
    #             noise = torch.randn_like(batch) * sigma
    #             predictions = self.model(batch + noise).argmax(1)
    #             counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
    #         return counts
        
    # def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
    #     counts = np.zeros(length, dtype=int)
    #     for idx in arr:
    #         counts[idx] += 1
    #     return counts

    # def smooth_predict(self, x, n=1000, alpha=0.1, batch_size=32):
    #     self.model.eval()
    #     counts = self._sample_noise(x, n, batch_size)
    #     top2 = counts.argsort()[::-1][:2]
    #     count1 = counts[top2[0]]
    #     count2 = counts[top2[1]]
    #     if binomtest(count1, count1 + count2, p=0.5).pvalue > alpha:
    #         return -1
    #     else:
    #         return top2[0]