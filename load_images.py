import os
import torch

from PIL import Image
from os import listdir
from typing import Tuple, List
from os.path import isfile, join
from torchvision import transforms


def load_content_and_style_paths(content_dir_path: str, style_dir_path: str) -> Tuple[List, List]:
    if not os.path.exists(content_dir_path):
        raise Exception(fr'The following path: {content_dir_path}, doesn\'t exist!')
    elif not os.path.exists(style_dir_path):
        raise Exception(fr'The following path: {style_dir_path}, doesn\'t exist!')

    content_images_full_path = [os.path.join(content_dir_path, im_name) for im_name in listdir(content_dir_path)
                                if isfile(join(content_dir_path, im_name)) and im_name.split('.')[-1] != 'db']
    style_images_full_path = [os.path.join(style_dir_path, im_name) for im_name in listdir(style_dir_path)
                              if isfile(join(style_dir_path, im_name)) and im_name.split('.')[-1] != 'db']

    return content_images_full_path, style_images_full_path


def _resize_image(image: Image, max_image_size, image_shape) -> torch.Tensor:
    # Clamping image size
    resize = max_image_size if max(image.size) > max_image_size else max(image.size)

    # Converting image size to image_shape should it be mentioned
    image_size = image_shape if image_shape is not None else (resize, resize)

    # Mean and std of ImageNet data set
    # This is rather important because the Vgg19 network (in PyTorch) was pre-trained on a classification task using
    # the ImageNet dataset. Hence, during the training process, all the images were to be normalized before passing them
    # to the model
    mean = [0.485, 0.456, 0.406]  # Mean of ImageNet data set
    std = [0.229, 0.224, 0.225]  # STD of ImageNet data set

    normalize = transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std))
    interpolation_mode = transforms.InterpolationMode.BICUBIC

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=interpolation_mode),
        transforms.ToTensor(),
        normalize])

    # discard the transparent (should it exists), alpha channel (that's the :3), and adding the batch dimension
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image


def load_image(path: str, max_image_size=500, image_shape=None) -> torch.Tensor:
    if not os.path.exists(path):
        dir_name = path.split('\\')[-2]
        path_dir: List = path.split('\\')[:-1]
        path_dir: str = '\\'.join(path_dir)
        raise Exception(f'In the path: {path_dir}, there is no such directory called: {dir_name}')

    image = Image.open(path).convert('RGB')
    resized_image = _resize_image(image, max_image_size, image_shape)

    return resized_image
