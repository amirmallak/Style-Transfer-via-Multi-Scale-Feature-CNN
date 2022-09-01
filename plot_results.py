import os
import torch
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple
from utilities import tensor_to_numpy


def plot_results(content_images_list: List[torch.Tensor], style_images_list: List[torch.Tensor],
                 target_images: Dict[str, List[Tuple[int, torch.Tensor]]], loss_list_f: List[List]) -> None:

    results_path = os.path.join('.', 'results')
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    for target_name, content_image, style_image, loss_list in zip(target_images.keys(), content_images_list,
                                                                  style_images_list, loss_list_f):
        content_image = tensor_to_numpy(content_image)
        style_image = tensor_to_numpy(style_image)

        target_path = os.path.join(results_path, target_name)

        target_iter_list: List = target_images[target_name]
        final_target_image = tensor_to_numpy(target_iter_list[-1][1])
        total_steps = len(loss_list)

        plt.figure()
        plt.rcParams.update({'font.size': 4})
        plt.suptitle(f'{target_name} results')
        plt.tight_layout()

        # --- First row ---

        ax_0_0 = plt.subplot(3, 3, 1)
        ax_0_0.imshow(content_image)
        ax_0_0.set_title('Content Image')

        ax_0_1 = plt.subplot(3, 3, 2)
        ax_0_1.imshow(style_image)
        ax_0_1.set_title('Style Image')

        ax_0_2 = plt.subplot(3, 3, 3)
        ax_0_2.imshow(final_target_image)
        ax_0_2.set_title('Target Image')

        # --- Second row ---

        num_rows = 3
        num_cols = len(target_iter_list)
        for i in range(num_cols):
            step = target_iter_list[i][0]
            intermediate_target_image = tensor_to_numpy(target_iter_list[i][1])

            # Iterate through the columns in the second row
            ax = plt.subplot(num_rows, num_cols, num_cols + (i+1))
            ax.imshow(intermediate_target_image)
            ax.set_title(f'Step = {step}/{total_steps}')

        # --- Third row ---

        ax_2_0 = plt.subplot(3, 1, 3)
        ax_2_0.plot(loss_list)
        ax_2_0.set_title('Total Loss graph')

        plt.savefig(target_path)
        # plt.show()
