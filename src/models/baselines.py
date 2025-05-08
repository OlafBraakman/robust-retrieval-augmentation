import timm
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, InterpolationMode
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from scipy.stats import binomtest
import numpy as np
from math import ceil


def get_adversarial_trained_vit_and_preprocessor(checkpoint_path, device="cuda"):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create the model instance
    model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=False).to(device)

    converted_state_dict = {
        (key[len("base_model."):]) if key.startswith("base_model.") else key: value 
        for key, value in checkpoint.items()
    }

    model.load_state_dict(converted_state_dict, strict=True)
    model.eval()

    t = []
    t.append(lambda x: Image.fromarray(x))
    t.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
    t.append(CenterCrop(224))
    t.append(ToTensor())

    return model, Compose(t)

def get_adversarial_trained_convnet_and_preprocessor(checkpoint_path, device="cuda"):
    model = timm.models.convnext.convnext_tiny(pretrained=True)
    model.stem = ConvBlock1(48, end_siz=8)

    model = model.to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    checkpoint = {k.replace('base_model.', ''): v for k, v in checkpoint.items()}
    checkpoint = {k.replace('se_', 'se_module.'): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    model.eval()

    t = []
    t.append(lambda x: Image.fromarray(x))
    t.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
    t.append(CenterCrop(224))
    t.append(ToTensor())
    
    return model, Compose(t)

def get_randomized_smoothing_resnet_50_and_preprocessor(checkpoint_path, device="cuda"):
    model = get_resnet().to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])


    model = Smooth(model, 1000, 0.5, 32, 0.001, 8)

    t = []
    t.append(lambda x: Image.fromarray(x))
    t.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
    t.append(CenterCrop(224))
    t.append(ToTensor())

    return model.to(device), Compose(t)

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds, device="cuda"):
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

def get_normalize_layer(device="cuda"):
    """Return the dataset's normalization layer"""
    return NormalizeLayer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device=device)

def get_resnet(device="cuda"):
    model = torch.nn.DataParallel(resnet50(pretrained=False)).to(device)

    return torch.nn.Sequential(get_normalize_layer(device=device), model)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class ConvBlock1(nn.Module):
    def __init__(self, siz=48, end_siz=8, fin_dim=384):
        super(ConvBlock1, self).__init__()
        self.planes = siz

        fin_dim = self.planes*end_siz if fin_dim == None else 432
        self.stem = nn.Sequential(nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes, self.planes*2, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*2, data_format="channels_first"),
                                  nn.GELU()
                                  )

    def forward(self, x):
        out = self.stem(x)
        # out = self.bn(out)
        return out

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
            # return Smooth.ABSTAIN # Commented for accuracy metric cannot be negative
            return top2[0]
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