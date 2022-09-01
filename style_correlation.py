import torch

from typing import Dict


def correlation_hermitian_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the correlation in between the local spatial features of a given feature map spatial wise,
    i.e. height and width wise, with respect to its hermitian matrix of inner product.

    It's important for the loss representation of the style features to be expressed as a matrix correlation with non
    spatial dependency (we're hitting a style representation. By mean, the style images high frequencies which
    represents texture, color, and style semantic objects shape. So we're not interested in a local spatial image
    features, but rather, a spanning-base tensor which holds this general style components).

    P.S. We still want the above Hermitial matrix representation to hold even when the image is shuffled in space (The
    style - color, texture, semantic shapes\and high frequencies, will still be there). That's why we're looking for a
    non-spatial matrix objective

    :param
    feature_map: (tensor) A feature map.
                 Shape: (B, C, H, W)

    :return:
    corr_hermit_matrix: (tensor) The correlated non-spatial Hermitian matrix of the input feature map.
                        Shape: (B*C, B*C)
    """

    # Extract the feature map's dimensions
    b, d, h, w = feature_map.size()

    # Reshaping for a Hermitian non-spatial representation
    spatial_tensor = feature_map.view(b * d, h * w)  # Shape: (B*C, H*W)

    # Correlation Hermitian non-spatial matrix
    corr_hermit_matrix = torch.mm(spatial_tensor, spatial_tensor.t())  # Shape: (B*C, B*C)

    return corr_hermit_matrix


def multi_scale_hermitian_matrix(feature_map_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    hermit_features: Dict[str, torch.Tensor] = {layer: correlation_hermitian_matrix(feature_map_dict[layer])
                                                for layer in feature_map_dict.keys()}

    return hermit_features
