import os
import torch
import cv2.cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import List


def image_read(paths: List[str]) -> List[np.ndarray]:
    images = []
    for path in paths:
        if not os.path.exists(path):
            dir_name = path.split('\\')[-2]
            path_dir: List = path.split('\\')[:-1]
            path_dir: str = '\\'.join(path_dir)
            raise Exception(f'In the path: {path_dir}, there is no such directory called: {dir_name}')

        images.append(cv2.imread(path))

    return images


def extract_target_name(content_path: str, style_path: str) -> str:
    content_name = content_path.split('.')[1].split('\\')[-1]
    style_name = style_path.split('.')[1].split('\\')[-1]

    target_name = f'{content_name}_&_{style_name}'

    return target_name


def show_images(content_im: torch.Tensor, style_im: torch.Tensor) -> None:
    plt.figure(figsize=(40, 80))
    plt.suptitle(f'Content and Style images chosen')

    images = [content_im, style_im]

    for i, image_tensor in enumerate(images):
        image = image_tensor.to("cpu").clone().detach()

        # Canceling the Batch dimension
        image = image.numpy().squeeze()

        # Converting to shape: (H, W, C), for numpy display
        image = image.transpose(1, 2, 0)

        # Mean and std of ImageNet data set
        # This is rather important because the Vgg19 network (in PyTorch) was pre-trained on a classification task using
        # the ImageNet dataset. Hence, during the training process, all the images were to be normalized before passing
        # them to the model
        mean = [0.485, 0.456, 0.406]  # Mean of ImageNet data set
        std = [0.229, 0.224, 0.225]  # STD of ImageNet data set

        # Un-normalizing image values by the mean and std which were normalized before
        image = image * np.array(std) + np.array(mean)

        # Clipping image values just in case
        image = image.clip(0, 1)

        plt.subplot(1, 2, i+1)

        # Choosing the correct subplot image's title
        image_title = 'Content_Image' if not (i % 2) else 'Style_Image'

        plt.title(image_title)
        plt.imshow(image)

    # plt.show(block=False)
    # plt.pause(1)
    # plt.close('all')
    plt.show()


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    image = image.to("cpu").clone().detach()

    # Canceling the Batch dimension
    image = image.numpy().squeeze()

    # Converting to shape: (H, W, C), for numpy display
    image = image.transpose(1, 2, 0)

    # Mean and std of ImageNet data set
    # This is rather important because the Vgg19 network (in PyTorch) was pre-trained on a classification task using
    # the ImageNet dataset. Hence, during the training process, all the images were to be normalized before passing
    # them to the model
    mean = [0.485, 0.456, 0.406]  # Mean of ImageNet data set
    std = [0.229, 0.224, 0.225]  # STD of ImageNet data set

    # Un-normalizing image values by the mean and std which were normalized before
    image = image * np.array(std) + np.array(mean)

    # Clipping image values just in case
    image = image.clip(0, 1)

    return image
