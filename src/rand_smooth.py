import torch
from scipy.stats import binomtest
import numpy as np
from math import ceil

from config.config import combine_cfg
from data.datasets.build import build_dataset
from data.preprocessing import get_preprocessor
from torchmetrics.classification import Accuracy
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from PIL import Image

import torch
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, Subset

import timm

device = "cuda"

from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn

class Smooth(torch.nn.Module):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, n, alpha, batch_size):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        super(Smooth, self).__init__()

        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.n = n
        self.alpha = alpha
        self.batch_size = batch_size

    def predict(self, x: torch.tensor) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, self.n, self.batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binomtest(count1, count1 + count2, p=0.5).pvalue > self.alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]
        
    def forward(self, x):
        self.base_classifier.eval()
        return self._sample_noise_loss(x, self.n, self.batch_size)

    def _sample_noise_loss(self, x: torch.tensor, num: int, batch_size):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        # with torch.no_grad():
        logits = torch.zeros(1, self.num_classes, device=x.device)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device='cuda') * self.sigma
            logits += self.base_classifier(batch + noise).mean(0)
        return logits

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds

def get_normalize_layer():
    """Return the dataset's normalization layer"""
    return NormalizeLayer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_resnet():
    model = torch.nn.DataParallel(resnet50(pretrained=False)).to(device)

    return torch.nn.Sequential(get_normalize_layer(), model)



if __name__ == "__main__":
    model = get_resnet().to(device)

    checkpoint_path = '/project/models/checkpoint.pth'

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    model = Smooth(model, 1000, 0.5, 32, 0.001, 8)

    config = combine_cfg("/project/src/config/dinov2_imagenet.yml")
    config['dataset']['name'] = "imagenet_classification"
    config['dataset']['dir'] = "/data/datasets/ImageNet"
    # config['dataset']['image_tag'] = ['image', 'dinov2_large', 'classification']
    # config['dataset']['depth_tag'] = ['depth', 'dinov2_large', 'classification']

    config['dataset']['split'] = 'test'
    val_dataset_base = build_dataset(config['dataset'])

    val_dataset = val_dataset_base

    t = []
    t.append(lambda x: Image.fromarray(x))
    t.append(Resize(224, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(CenterCrop(224))
    t.append(ToTensor())

    val_dataset.preprocessor = Compose(t)
    # get_preprocessor(
    #     val_dataset,
    #     height=config['model']['backbone']['height'],
    #     width=config['model']['backbone']['width'],
    #     phase="test",
    # )
    # subset_indices = random.sample(list(range(len(val_dataset))), k=250)
    # subset_dataset = Subset(val_dataset, subset_indices)
    dataloader = DataLoader(val_dataset, batch_size=1)

    original_accuracy = Accuracy(task='multiclass', num_classes=1000, average='micro').to(device)
    # original_accuracy_macro = Accuracy(task='multiclass', num_classes=1000, average='macro').to(device)
    adversarial_accuracy = Accuracy(task='multiclass', num_classes=1000, average='micro').to(device)
    # adversarial_accuracy_macro = Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(args.device)

    # correct = 0
    # num_samples = 0

    for (X,y), _ in tqdm(dataloader):
        X = X.float().to(device)
        y = y.to(device)

        # correct += (get_pred(model, X, args.device) == y.cpu()).sum()
        # num_samples += len(X)

        pred = model.predict(X)
        original_accuracy(torch.tensor(pred).unsqueeze(0).to(device), y)
    
    print(original_accuracy.compute().item())