import time
import torch
import random
import torch.optim as optim

from tqdm import tqdm
from typing import Dict
from plot_results import plot_results
from load_model import load_vgg19_model
from extract_feature_maps import extract_feature_maps
from utilities import show_images, extract_target_name
from style_correlation import multi_scale_hermitian_matrix
from hyperparameters_weights import loss_and_hermit_weights
from load_images import load_content_and_style_paths, load_image


def multi_scale_style_transfer(content_dir_path, style_dir_path, number_of_content_and_style_images) -> None:

    # Pick a device to run on
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print(f'The code is running on: {device} (Device)\n')

    # Loading our relevant model
    vgg19_features = load_vgg19_model()

    # Loading our model to device
    vgg19_features.to(device=device)

    # Print Neural Network layers (To define the mapping of each layer later in code)
    print(vgg19_features)

    # Loading the content and style images paths
    content_paths, style_paths = load_content_and_style_paths(content_dir_path, style_dir_path)

    # Picking up random content and style images out of all the content and style images, individually
    content_random_index = random.sample(range(0, len(content_paths)), number_of_content_and_style_images)
    selected_content_images = [content_paths[index] for index in content_random_index]
    style_random_index = random.sample(range(0, len(style_paths)), number_of_content_and_style_images)
    selected_style_images = [style_paths[index] for index in style_random_index]

    t_start = time.time()
    intermediate_images = {}
    content_images_list = []
    style_images_list = []
    total_loss_list_f = []
    for content_image_path, style_image_path in zip(selected_content_images, selected_style_images):
        total_loss_list = []

        # Building Target Image name and mapping it to an empty list
        target_name: str = extract_target_name(content_image_path, style_image_path)
        intermediate_images[target_name] = []

        # Loading the content image as a torch Tensor
        # Image shape: (B, C, H, W)
        content_image: torch.Tensor = load_image(content_image_path).to(device)

        content_images_list.append(content_image.clone())

        # Loading the style image as a torch Tensor. And resizing it as the content one
        # shape[-2:] - Picking up the H and W dimensions
        style_image: torch.Tensor = load_image(style_image_path, image_shape=content_image.shape[-2:]).to(device)

        style_images_list.append(style_image.clone())

        # Display both images
        show_images(content_image, style_image)

        """ Extracting Features, Hermitian matrices, and Weights """

        # Extracting content desired feature map via Vgg19
        content_feature_map: Dict[str:torch.Tensor] = extract_feature_maps(content_image, content_style_target=0,
                                                                           model=vgg19_features)

        # Extracting style desired feature maps via Vgg19
        style_feature_map: Dict[str:torch.Tensor] = extract_feature_maps(style_image, content_style_target=1,
                                                                         model=vgg19_features)

        # Calculating the Correlated Hermitian non-spatial matrix
        style_hermit_features: Dict[str:torch.Tensor] = multi_scale_hermitian_matrix(style_feature_map)

        # Content, Style Loss weight and Style inner Hermitian matrices weights
        content_style_params = loss_and_hermit_weights(style_feature_map.keys(), content_feature_map.keys())
        content_loss_weight, style_loss_weight, content_hermit_weights, style_hermit_weights = content_style_params

        # Creating a target image which, initially, stated as the content image
        target_image = content_image.clone()

        # Setting the target image to an initial Gaussian Random Noise
        # target_image = torch.randn(content_image.shape).to(device=device)

        # Changing the target image over the training course
        target_image.requires_grad_(True).to(device=device)

        """ Creating a Style Transferred Image """

        fine_tuning = 4e3
        optimizer: torch.optim = optim.Adam([target_image], lr=3e-3)

        print('\nStarting Target Image Fine Tuning! ...\n')

        for step in tqdm(range(1, int(fine_tuning)+1)):

            # Extracting target image desired feature maps via Vgg19
            target_feature_map: Dict[str:torch.Tensor] = extract_feature_maps(target_image, content_style_target=2,
                                                                              model=vgg19_features)

            # Calculating target image Hermitian non-spatial correlated matrices
            target_hermit_features: Dict[str:torch.Tensor] = multi_scale_hermitian_matrix(target_feature_map)

            target_content_loss = torch.zeros(1).to(device=device)
            for layer_name in content_feature_map.keys():
                # Extracting relevant feature map (according to each desired target layer in vgg19)
                target_tensor: torch.Tensor = target_feature_map[layer_name]

                # Extracting relevant feature map (according to each desired content layer in vgg19)
                content_tensor = content_feature_map[layer_name]

                # Calculating the Target Loss with respect to the Content Features
                target_content_layer_loss = torch.pow((target_tensor - content_tensor), 2)
                target_content_layer_loss = torch.mean(target_content_layer_loss)
                target_content_layer_loss *= content_hermit_weights[layer_name]

                target_content_loss += target_content_layer_loss

            target_style_loss = torch.zeros(1).to(device=device)
            for layer_name in style_feature_map.keys():
                # Extracting relevant feature map (according to each desired target layer in vgg19)
                target_tensor: torch.Tensor = target_feature_map[layer_name]

                # Extracting target feature maps dimensions
                b, c, h, w = target_tensor.shape

                # Extracting target relevant Hermitian matrix (according to each desired target layer in vgg19)
                target_hermit_tensor: torch.Tensor = target_hermit_features[layer_name]

                # Extracting style relevant Hermitian matrix (according to each desired style layer in vgg19)
                style_hermit_tensor: torch.Tensor = style_hermit_features[layer_name]

                # Calculating the Target Loss with respect to the Style Hermitian Features
                target_style_layer_loss = torch.pow((target_hermit_tensor - style_hermit_tensor), 2)
                target_style_layer_loss = torch.mean(target_style_layer_loss)
                target_style_layer_loss *= style_hermit_weights[layer_name]

                target_style_loss += target_style_layer_loss / (b * c * h * w)

            # Calculating total loss
            total_loss = content_loss_weight * target_content_loss + style_loss_weight * target_style_loss
            total_loss_list.append(total_loss.item())

            # Fine tuning Target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Collecting intermediate target images and printing the total loss
            if not (step % 8e2):
                print(f'Step: {step}/{int(fine_tuning)}, Total Loss: {total_loss.item() / 1e6}x1e6')
                intermediate_images[target_name].append((step, target_image.clone()))

        total_loss_list_f.append(total_loss_list)
    plot_results(content_images_list, style_images_list, intermediate_images, total_loss_list_f)

    t_end = time.time()
    hours = int((t_end - t_start) / 3.6e3)
    minutes = int((((t_end - t_start) / 3.6e3) - hours) * 60)

    print(f'\nTotal model style transfer run time is: {hours}:{minutes}[h]\n')
