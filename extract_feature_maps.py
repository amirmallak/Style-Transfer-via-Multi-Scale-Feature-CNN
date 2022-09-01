import torch
import torchvision

from typing import Dict


def extract_feature_maps(image: torch.Tensor, content_style_target: int, model: torchvision.models,
                         vgg19_extraction_layers: Dict[str, str] = None) -> Dict[str, torch.Tensor]:

    # In case the user didn't specify any special desirable model layers to extract features out of
    if not vgg19_extraction_layers:
        # Defining the default layers in which to extract the feature maps from
        # The layers are defined as follows: 'key' - actual layer number in the vgg19 network (could be found when
        # printed earlier). 'value' - a more interpretable name for this specific layer

        # If it's a content image
        if content_style_target == 0:
            vgg19_extraction_layers = {'12': 'conv3_2',
                                       '21': 'conv4_2'}

        # If it's a style image
        elif content_style_target == 1:
            vgg19_extraction_layers = {'0': 'conv1_1',
                                       '5': 'conv2_1',
                                       '10': 'conv3_1',
                                       '19': 'conv4_1',
                                       '28': 'conv5_1'}

        # If it's a target image
        elif content_style_target == 2:
            vgg19_extraction_layers = {'0': 'conv1_1',
                                       '5': 'conv2_1',
                                       '10': 'conv3_1',
                                       '12': 'conv3_2',
                                       '19': 'conv4_1',
                                       '21': 'conv4_2',
                                       '28': 'conv5_1'}

    feature_maps = {}
    feature_map = image

    # Running the input image (content\style\target) through the network and extracting desirable feature maps
    for layer_name, model_layer in model._modules.items():
        feature_map = model_layer(feature_map)

        # If the model's layer is one which interests us for our research
        if layer_name in vgg19_extraction_layers:
            interpretable_name = vgg19_extraction_layers[layer_name]
            feature_maps[interpretable_name] = feature_map

    return feature_maps
