# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

def resize(image, temperature, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    # print(size)
    rescaled_image = F.resize(image, size)
    rescaled_temperature = F.resize(temperature, size)

    return rescaled_image, rescaled_temperature


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, temperature):
        size = random.choice(self.sizes)
        return resize(img, temperature, size, self.max_size)



class ToTensor(object):
    def __call__(self, img, temperature):
        return F.to_tensor(img), F.to_tensor(temperature)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, temperature):
        image = F.normalize(image, mean=self.mean, std=self.std)
        temperature = F.normalize(temperature, mean=self.mean, std=self.std)
        return image, temperature


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, temperature):
        for t in self.transforms:
            image, temperature = t(image, temperature)
        return image, temperature

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
