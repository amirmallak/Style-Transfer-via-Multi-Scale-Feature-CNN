from torchvision import models


def load_vgg19_model():
    vgg19_model = models.vgg19(pretrained=True, progress=True)

    # For the style transfer research we'll only need the Features part of the Vgg19 Network (which contains the CNN
    # portion without the dense layers - which were used for the final classification task on ImageNet)
    vgg19_features = vgg19_model.features

    # Freezing the network params (keeping the network static for the whole process)
    for param in vgg19_features.parameters():
        # param.requires_grad = False
        param.requires_grad_(False)

    return vgg19_features
