from typing import List, Dict, Tuple


def loss_and_hermit_weights(style_layer_names: Dict.keys = None, content_layer_names: Dict.keys = None,
                            content_hermit_weights_list: List[float] = None,
                            style_hermit_weights_list: List[float] = None) -> \
                            Tuple[float, float, Dict[str, float], Dict[str, float]]:

    content_loss_weight = 1
    # content_loss_weight = 1e6

    style_loss_weight = 1e6
    # style_loss_weight = 5e5

    if not content_hermit_weights_list:
        # content_hermit_weights_list = [1]
        content_hermit_weights_list = [9e-1, 1]

    if not style_hermit_weights_list:
        style_hermit_weights_list = [1., 75e-2, 2e-1, 2e-1, 2e-1]
        # style_hermit_weights_list = [9e-1, 75e-2, 6e-1, 6e-1, 6e-1]

    content_hermit_weights = {layer_name: weight for layer_name, weight in zip(content_layer_names,
                                                                               content_hermit_weights_list)}
    style_hermit_weights = {layer_name: weight for layer_name, weight in zip(style_layer_names,
                                                                             style_hermit_weights_list)}

    return content_loss_weight, style_loss_weight, content_hermit_weights, style_hermit_weights
