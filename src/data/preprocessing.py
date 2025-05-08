# Modified from: https://github.com/Barchid/RGBD-Seg/blob/master/preprocessing.py
import cv2
import matplotlib
import matplotlib.colors
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from data.datasets.dataset_base import DatasetBase, DatasetBaseDepth

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def get_preprocessor(dataset,
                     height=None,
                     width=None,
                     phase='train'):
    # if isinstance(dataset, DatasetBaseDepth):
    #     return get_rgbd_preprocessor(
    #         dataset.depth_mean,
    #         dataset.depth_std,
    #         dataset.depth_mode,
    #         height=height,
    #         width=width,
    #         phase=phase
    #     )
    # elif isinstance(dataset, DatasetBase):
    return get_rgb_preprocessor(
        height=height,
        width=width,
        phase=phase
    )
    # else:
    #     raise Exception("Dataset not a subclass of DatasetBase")

def get_rgb_preprocessor(height=None,
                     width=None,
                     phase='train'):

    assert phase in ['train', 'test']

    transform_list = [Rescale(height, width)]

    transform_list.extend([
        ToTensor(),
        Normalize(),
    ])
        
    transform = transforms.Compose(transform_list)
    return transform

def get_rgbd_preprocessor(depth_mean,
                     depth_std,
                     depth_mode='refined',
                     height=None,
                     width=None,
                     phase='train'):
    assert phase in ['train', 'test']

    # if phase == 'train':
    #         transform_list = [
    #         # Rescale(int(height*1.4), int(width*1.4)),
    #         # RandomCrop(crop_height=height, crop_width=width),
    #         # RandomHSV((0.9, 1.1),
    #         #           (0.9, 1.1),
    #         #           (25, 25)),
    #         # RandomFlip(),
    #         # ToTensor(),
    #         # Normalize(depth_mean=depth_mean,
    #         #           depth_std=depth_std,
    #         #           depth_mode=depth_mode),
    #     ]
    # else:
    #     if height is None and width is None:
    #         transform_list = []
    #     else:
    transform_list = [Rescale(height=height, width=width)]
        # transform_list.extend([
        #     ToTensor(),
        #     Normalize(depth_mean=depth_mean,
        #               depth_std=depth_std,
        #               depth_mode=depth_mode)
        # ])
    transform = transforms.Compose(transform_list)
    return transform


class Rescale:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        return cv2.resize(sample, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

class ToTensor:
    def __call__(self, sample):
        sample = sample.transpose((2, 0, 1))
        return torch.from_numpy(sample).float() / 255.

class Normalize:

    def __call__(self, sample):
        # image, depth = sample['image'], sample['depth']

        sample = torchvision.transforms.Normalize(
            mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)(sample)

        return sample


# class Rescale:
#     def __init__(self, height, width):
#         self.height = height
#         self.width = width

#     def __call__(self, sample):
#         if "image" in sample:
#             image = sample["image"]
#             image = cv2.resize(image, (self.width, self.height),
#                             interpolation=cv2.INTER_LINEAR)
#             sample['image'] = image

#         if "depth" in sample:
#             depth = sample["depth"]
#             depth = cv2.resize(depth, (self.width, self.height),
#                             interpolation=cv2.INTER_NEAREST)
#             sample['depth'] = depth

#         return sample
    

# class RandomRescale:
#     def __init__(self, scale):
#         self.scale_low = min(scale)
#         self.scale_high = max(scale)

#     def __call__(self, sample):
#         target_scale = np.random.uniform(self.scale_low, self.scale_high)

#         if "image" in sample:
#             image = sample["image"]
#             target_height = int(round(target_scale * image.shape[0]))
#             target_width = int(round(target_scale * image.shape[1]))

#             image = cv2.resize(image, (target_width, target_height),
#                             interpolation=cv2.INTER_LINEAR)
#             sample['image'] = image

#         if "depth" in sample:
#             depth = sample["depth"]
#             target_height = int(round(target_scale * depth.shape[0]))
#             target_width = int(round(target_scale * depth.shape[1]))
#             depth = cv2.resize(depth, (target_width, target_height),
#                             interpolation=cv2.INTER_NEAREST)
#             sample['depth'] = depth

#         return sample


# class RandomCrop:
#     def __init__(self, crop_height, crop_width):
#         self.crop_height = crop_height
#         self.crop_width = crop_width
#         self.rescale = Rescale(self.crop_height, self.crop_width)

#     def __call__(self, sample):
#         if "image" in sample: 
#             image = sample["image"]
#             h = image.shape[0]
#             w = image.shape[1]
#             if h <= self.crop_height or w <= self.crop_width:
#                 # simply rescale instead of random crop as image is not large enough
#                 sample = self.rescale(sample)
#             else:
#                 i = np.random.randint(0, h - self.crop_height)
#                 j = np.random.randint(0, w - self.crop_width)
#                 image = image[i:i + self.crop_height, j:j + self.crop_width, :]
#                 sample['image'] = image

#         if "depth" in sample:
#             depth = sample["image"]
#             h = depth.shape[0]
#             w = depth.shape[1]
#             if h <= self.crop_height or w <= self.crop_width:
#                 # simply rescale instead of random crop as image is not large enough
#                 sample = self.rescale(sample)
#             else:
#                 i = np.random.randint(0, h - self.crop_height)
#                 j = np.random.randint(0, w - self.crop_width)
#                 depth = depth[i:i + self.crop_height, j:j + self.crop_width]

#                 sample['depth'] = depth

#         return sample


# class RandomHSV:
#     def __init__(self, h_range, s_range, v_range):
#         assert isinstance(h_range, (list, tuple)) and \
#                isinstance(s_range, (list, tuple)) and \
#                isinstance(v_range, (list, tuple))
#         self.h_range = h_range
#         self.s_range = s_range
#         self.v_range = v_range

#     def __call__(self, sample):

#         if "image" in sample:
#             img = sample['image']
#             img_hsv = matplotlib.colors.rgb_to_hsv(img)
#             img_h = img_hsv[:, :, 0]
#             img_s = img_hsv[:, :, 1]
#             img_v = img_hsv[:, :, 2]

#             h_random = np.random.uniform(min(self.h_range), max(self.h_range))
#             s_random = np.random.uniform(min(self.s_range), max(self.s_range))
#             v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
#             img_h = np.clip(img_h * h_random, 0, 1)
#             img_s = np.clip(img_s * s_random, 0, 1)
#             img_v = np.clip(img_v + v_random, 0, 255)
#             img_hsv = np.stack([img_h, img_s, img_v], axis=2)
#             img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

#             sample['image'] = img_new

#         return sample


# class RandomFlip:
#     def __call__(self, sample):
#         if np.random.rand() > 0.5:
#             if "image" in sample:
#                 image = sample["image"]
#                 image = np.fliplr(image).copy()    
#                 sample['image'] = image        
#             if "depth" in sample:
#                 depth = sample["depth"]    
#                 depth = np.fliplr(depth).copy()
#                 sample['depth'] = depth
#         return sample


# class Normalize:
#     def __init__(self, depth_mean=None, depth_std=None, depth_mode='refined'):
#         assert depth_mode in ['refined', 'raw']
#         self._depth_mode = depth_mode
#         self._depth_mean = [depth_mean]
#         self._depth_std = [depth_std]

#     def __call__(self, sample):
#         # image, depth = sample['image'], sample['depth']

#         if "image" in sample:
#             image = sample['image']
#             image = image / 255
#             image = torchvision.transforms.Normalize(
#                 mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)(image)
            
#             sample['image'] = image

#         if "depth" in sample:
#             depth = sample['depth']
#             if self._depth_mode == 'raw':
#                 depth_0 = depth == 0
#                 depth = torchvision.transforms.Normalize(
#                     mean=self._depth_mean, std=self._depth_std)(depth)
#                 depth[depth_0] = 0
#             else:
#                 depth = torchvision.transforms.Normalize(
#                     mean=self._depth_mean, std=self._depth_std)(depth)

#             sample['depth'] = depth

#         return sample


# class ToTensor:
#     def __call__(self, sample):

#         if "image" in sample:
#             image = sample["image"]
#             image = image.transpose((2, 0, 1))
#             sample['image'] = torch.from_numpy(image).float()

#         if "depth" in sample:
#             depth = sample["depth"]
#             depth = np.expand_dims(depth, 0).astype('float32')
#             sample['depth'] = torch.from_numpy(depth).float()

#         return sample
