import numpy as np
import torch
import torch.nn.functional as F


import torch


def me():
    print("me")


kou = None
kai = ""


if {"a": "b"}:
    print("mymy")


# x = torch.arange(0, 7, 1).view(-1, 1).repeat(1, 7)
# print(x)
# print(x.sum(dim=1))
# # print(x.shape[-3])
# y = torch.arange(1, 5, 1).view(-1, 1, 1)
# z = x * y
# print(
#     F.normalize(
#         torch.tensor(np.array([[1, 5, 10], [5, 25, 50]]), dtype=torch.float32), dim=0
#     )
# )


# discard "module.encoder_k/projector_k.*" "module.projector.*" and "module.value_transform.*"
lis = [
    "module.encoder.conv1.weight",
    "module.encoder.bn1.weight",
    "module.encoder.bn1.bias",
    "module.encoder.bn1.running_mean",
    "module.encoder.bn1.running_var",
    "module.encoder.bn1.num_batches_tracked",
    "module.encoder.layer1.0.conv1.weight",
    "module.encoder.layer1.0.bn1.weight",
    "module.encoder.layer1.0.bn1.bias",
    "module.encoder.layer1.0.bn1.running_mean",
    "module.encoder.layer1.0.bn1.running_var",
    "module.encoder.layer1.0.bn1.num_batches_tracked",
    "module.encoder.layer1.0.conv2.weight",
    "module.encoder.layer1.0.bn2.weight",
    "module.encoder.layer1.0.bn2.bias",
    "module.encoder.layer1.0.bn2.running_mean",
    "module.encoder.layer1.0.bn2.running_var",
    "module.encoder.layer1.0.bn2.num_batches_tracked",
    "module.encoder.layer1.0.conv3.weight",
    "module.encoder.layer1.0.bn3.weight",
    "module.encoder.layer1.0.bn3.bias",
    "module.encoder.layer1.0.bn3.running_mean",
    "module.encoder.layer1.0.bn3.running_var",
    "module.encoder.layer1.0.bn3.num_batches_tracked",
    "module.encoder.layer1.0.downsample.0.weight",
    "module.encoder.layer1.0.downsample.1.weight",
    "module.encoder.layer1.0.downsample.1.bias",
    "module.encoder.layer1.0.downsample.1.running_mean",
    "module.encoder.layer1.0.downsample.1.running_var",
    "module.encoder.layer1.0.downsample.1.num_batches_tracked",
    "module.encoder.layer1.1.conv1.weight",
    "module.encoder.layer1.1.bn1.weight",
    "module.encoder.layer1.1.bn1.bias",
    "module.encoder.layer1.1.bn1.running_mean",
    "module.encoder.layer1.1.bn1.running_var",
    "module.encoder.layer1.1.bn1.num_batches_tracked",
    "module.encoder.layer1.1.conv2.weight",
    "module.encoder.layer1.1.bn2.weight",
    "module.encoder.layer1.1.bn2.bias",
    "module.encoder.layer1.1.bn2.running_mean",
    "module.encoder.layer1.1.bn2.running_var",
    "module.encoder.layer1.1.bn2.num_batches_tracked",
    "module.encoder.layer1.1.conv3.weight",
    "module.encoder.layer1.1.bn3.weight",
    "module.encoder.layer1.1.bn3.bias",
    "module.encoder.layer1.1.bn3.running_mean",
    "module.encoder.layer1.1.bn3.running_var",
    "module.encoder.layer1.1.bn3.num_batches_tracked",
    "module.encoder.layer1.2.conv1.weight",
    "module.encoder.layer1.2.bn1.weight",
    "module.encoder.layer1.2.bn1.bias",
    "module.encoder.layer1.2.bn1.running_mean",
    "module.encoder.layer1.2.bn1.running_var",
    "module.encoder.layer1.2.bn1.num_batches_tracked",
    "module.encoder.layer1.2.conv2.weight",
    "module.encoder.layer1.2.bn2.weight",
    "module.encoder.layer1.2.bn2.bias",
    "module.encoder.layer1.2.bn2.running_mean",
    "module.encoder.layer1.2.bn2.running_var",
    "module.encoder.layer1.2.bn2.num_batches_tracked",
    "module.encoder.layer1.2.conv3.weight",
    "module.encoder.layer1.2.bn3.weight",
    "module.encoder.layer1.2.bn3.bias",
    "module.encoder.layer1.2.bn3.running_mean",
    "module.encoder.layer1.2.bn3.running_var",
    "module.encoder.layer1.2.bn3.num_batches_tracked",
    "module.encoder.layer2.0.conv1.weight",
    "module.encoder.layer2.0.bn1.weight",
    "module.encoder.layer2.0.bn1.bias",
    "module.encoder.layer2.0.bn1.running_mean",
    "module.encoder.layer2.0.bn1.running_var",
    "module.encoder.layer2.0.bn1.num_batches_tracked",
    "module.encoder.layer2.0.conv2.weight",
    "module.encoder.layer2.0.bn2.weight",
    "module.encoder.layer2.0.bn2.bias",
    "module.encoder.layer2.0.bn2.running_mean",
    "module.encoder.layer2.0.bn2.running_var",
    "module.encoder.layer2.0.bn2.num_batches_tracked",
    "module.encoder.layer2.0.conv3.weight",
    "module.encoder.layer2.0.bn3.weight",
    "module.encoder.layer2.0.bn3.bias",
    "module.encoder.layer2.0.bn3.running_mean",
    "module.encoder.layer2.0.bn3.running_var",
    "module.encoder.layer2.0.bn3.num_batches_tracked",
    "module.encoder.layer2.0.downsample.0.weight",
    "module.encoder.layer2.0.downsample.1.weight",
    "module.encoder.layer2.0.downsample.1.bias",
    "module.encoder.layer2.0.downsample.1.running_mean",
    "module.encoder.layer2.0.downsample.1.running_var",
    "module.encoder.layer2.0.downsample.1.num_batches_tracked",
    "module.encoder.layer2.1.conv1.weight",
    "module.encoder.layer2.1.bn1.weight",
    "module.encoder.layer2.1.bn1.bias",
    "module.encoder.layer2.1.bn1.running_mean",
    "module.encoder.layer2.1.bn1.running_var",
    "module.encoder.layer2.1.bn1.num_batches_tracked",
    "module.encoder.layer2.1.conv2.weight",
    "module.encoder.layer2.1.bn2.weight",
    "module.encoder.layer2.1.bn2.bias",
    "module.encoder.layer2.1.bn2.running_mean",
    "module.encoder.layer2.1.bn2.running_var",
    "module.encoder.layer2.1.bn2.num_batches_tracked",
    "module.encoder.layer2.1.conv3.weight",
    "module.encoder.layer2.1.bn3.weight",
    "module.encoder.layer2.1.bn3.bias",
    "module.encoder.layer2.1.bn3.running_mean",
    "module.encoder.layer2.1.bn3.running_var",
    "module.encoder.layer2.1.bn3.num_batches_tracked",
    "module.encoder.layer2.2.conv1.weight",
    "module.encoder.layer2.2.bn1.weight",
    "module.encoder.layer2.2.bn1.bias",
    "module.encoder.layer2.2.bn1.running_mean",
    "module.encoder.layer2.2.bn1.running_var",
    "module.encoder.layer2.2.bn1.num_batches_tracked",
    "module.encoder.layer2.2.conv2.weight",
    "module.encoder.layer2.2.bn2.weight",
    "module.encoder.layer2.2.bn2.bias",
    "module.encoder.layer2.2.bn2.running_mean",
    "module.encoder.layer2.2.bn2.running_var",
    "module.encoder.layer2.2.bn2.num_batches_tracked",
    "module.encoder.layer2.2.conv3.weight",
    "module.encoder.layer2.2.bn3.weight",
    "module.encoder.layer2.2.bn3.bias",
    "module.encoder.layer2.2.bn3.running_mean",
    "module.encoder.layer2.2.bn3.running_var",
    "module.encoder.layer2.2.bn3.num_batches_tracked",
    "module.encoder.layer2.3.conv1.weight",
    "module.encoder.layer2.3.bn1.weight",
    "module.encoder.layer2.3.bn1.bias",
    "module.encoder.layer2.3.bn1.running_mean",
    "module.encoder.layer2.3.bn1.running_var",
    "module.encoder.layer2.3.bn1.num_batches_tracked",
    "module.encoder.layer2.3.conv2.weight",
    "module.encoder.layer2.3.bn2.weight",
    "module.encoder.layer2.3.bn2.bias",
    "module.encoder.layer2.3.bn2.running_mean",
    "module.encoder.layer2.3.bn2.running_var",
    "module.encoder.layer2.3.bn2.num_batches_tracked",
    "module.encoder.layer2.3.conv3.weight",
    "module.encoder.layer2.3.bn3.weight",
    "module.encoder.layer2.3.bn3.bias",
    "module.encoder.layer2.3.bn3.running_mean",
    "module.encoder.layer2.3.bn3.running_var",
    "module.encoder.layer2.3.bn3.num_batches_tracked",
    "module.encoder.layer3.0.conv1.weight",
    "module.encoder.layer3.0.bn1.weight",
    "module.encoder.layer3.0.bn1.bias",
    "module.encoder.layer3.0.bn1.running_mean",
    "module.encoder.layer3.0.bn1.running_var",
    "module.encoder.layer3.0.bn1.num_batches_tracked",
    "module.encoder.layer3.0.conv2.weight",
    "module.encoder.layer3.0.bn2.weight",
    "module.encoder.layer3.0.bn2.bias",
    "module.encoder.layer3.0.bn2.running_mean",
    "module.encoder.layer3.0.bn2.running_var",
    "module.encoder.layer3.0.bn2.num_batches_tracked",
    "module.encoder.layer3.0.conv3.weight",
    "module.encoder.layer3.0.bn3.weight",
    "module.encoder.layer3.0.bn3.bias",
    "module.encoder.layer3.0.bn3.running_mean",
    "module.encoder.layer3.0.bn3.running_var",
    "module.encoder.layer3.0.bn3.num_batches_tracked",
    "module.encoder.layer3.0.downsample.0.weight",
    "module.encoder.layer3.0.downsample.1.weight",
    "module.encoder.layer3.0.downsample.1.bias",
    "module.encoder.layer3.0.downsample.1.running_mean",
    "module.encoder.layer3.0.downsample.1.running_var",
    "module.encoder.layer3.0.downsample.1.num_batches_tracked",
    "module.encoder.layer3.1.conv1.weight",
    "module.encoder.layer3.1.bn1.weight",
    "module.encoder.layer3.1.bn1.bias",
    "module.encoder.layer3.1.bn1.running_mean",
    "module.encoder.layer3.1.bn1.running_var",
    "module.encoder.layer3.1.bn1.num_batches_tracked",
    "module.encoder.layer3.1.conv2.weight",
    "module.encoder.layer3.1.bn2.weight",
    "module.encoder.layer3.1.bn2.bias",
    "module.encoder.layer3.1.bn2.running_mean",
    "module.encoder.layer3.1.bn2.running_var",
    "module.encoder.layer3.1.bn2.num_batches_tracked",
    "module.encoder.layer3.1.conv3.weight",
    "module.encoder.layer3.1.bn3.weight",
    "module.encoder.layer3.1.bn3.bias",
    "module.encoder.layer3.1.bn3.running_mean",
    "module.encoder.layer3.1.bn3.running_var",
    "module.encoder.layer3.1.bn3.num_batches_tracked",
    "module.encoder.layer3.2.conv1.weight",
    "module.encoder.layer3.2.bn1.weight",
    "module.encoder.layer3.2.bn1.bias",
    "module.encoder.layer3.2.bn1.running_mean",
    "module.encoder.layer3.2.bn1.running_var",
    "module.encoder.layer3.2.bn1.num_batches_tracked",
    "module.encoder.layer3.2.conv2.weight",
    "module.encoder.layer3.2.bn2.weight",
    "module.encoder.layer3.2.bn2.bias",
    "module.encoder.layer3.2.bn2.running_mean",
    "module.encoder.layer3.2.bn2.running_var",
    "module.encoder.layer3.2.bn2.num_batches_tracked",
    "module.encoder.layer3.2.conv3.weight",
    "module.encoder.layer3.2.bn3.weight",
    "module.encoder.layer3.2.bn3.bias",
    "module.encoder.layer3.2.bn3.running_mean",
    "module.encoder.layer3.2.bn3.running_var",
    "module.encoder.layer3.2.bn3.num_batches_tracked",
    "module.encoder.layer3.3.conv1.weight",
    "module.encoder.layer3.3.bn1.weight",
    "module.encoder.layer3.3.bn1.bias",
    "module.encoder.layer3.3.bn1.running_mean",
    "module.encoder.layer3.3.bn1.running_var",
    "module.encoder.layer3.3.bn1.num_batches_tracked",
    "module.encoder.layer3.3.conv2.weight",
    "module.encoder.layer3.3.bn2.weight",
    "module.encoder.layer3.3.bn2.bias",
    "module.encoder.layer3.3.bn2.running_mean",
    "module.encoder.layer3.3.bn2.running_var",
    "module.encoder.layer3.3.bn2.num_batches_tracked",
    "module.encoder.layer3.3.conv3.weight",
    "module.encoder.layer3.3.bn3.weight",
    "module.encoder.layer3.3.bn3.bias",
    "module.encoder.layer3.3.bn3.running_mean",
    "module.encoder.layer3.3.bn3.running_var",
    "module.encoder.layer3.3.bn3.num_batches_tracked",
    "module.encoder.layer3.4.conv1.weight",
    "module.encoder.layer3.4.bn1.weight",
    "module.encoder.layer3.4.bn1.bias",
    "module.encoder.layer3.4.bn1.running_mean",
    "module.encoder.layer3.4.bn1.running_var",
    "module.encoder.layer3.4.bn1.num_batches_tracked",
    "module.encoder.layer3.4.conv2.weight",
    "module.encoder.layer3.4.bn2.weight",
    "module.encoder.layer3.4.bn2.bias",
    "module.encoder.layer3.4.bn2.running_mean",
    "module.encoder.layer3.4.bn2.running_var",
    "module.encoder.layer3.4.bn2.num_batches_tracked",
    "module.encoder.layer3.4.conv3.weight",
    "module.encoder.layer3.4.bn3.weight",
    "module.encoder.layer3.4.bn3.bias",
    "module.encoder.layer3.4.bn3.running_mean",
    "module.encoder.layer3.4.bn3.running_var",
    "module.encoder.layer3.4.bn3.num_batches_tracked",
    "module.encoder.layer3.5.conv1.weight",
    "module.encoder.layer3.5.bn1.weight",
    "module.encoder.layer3.5.bn1.bias",
    "module.encoder.layer3.5.bn1.running_mean",
    "module.encoder.layer3.5.bn1.running_var",
    "module.encoder.layer3.5.bn1.num_batches_tracked",
    "module.encoder.layer3.5.conv2.weight",
    "module.encoder.layer3.5.bn2.weight",
    "module.encoder.layer3.5.bn2.bias",
    "module.encoder.layer3.5.bn2.running_mean",
    "module.encoder.layer3.5.bn2.running_var",
    "module.encoder.layer3.5.bn2.num_batches_tracked",
    "module.encoder.layer3.5.conv3.weight",
    "module.encoder.layer3.5.bn3.weight",
    "module.encoder.layer3.5.bn3.bias",
    "module.encoder.layer3.5.bn3.running_mean",
    "module.encoder.layer3.5.bn3.running_var",
    "module.encoder.layer3.5.bn3.num_batches_tracked",
    "module.encoder.layer4.0.conv1.weight",
    "module.encoder.layer4.0.bn1.weight",
    "module.encoder.layer4.0.bn1.bias",
    "module.encoder.layer4.0.bn1.running_mean",
    "module.encoder.layer4.0.bn1.running_var",
    "module.encoder.layer4.0.bn1.num_batches_tracked",
    "module.encoder.layer4.0.conv2.weight",
    "module.encoder.layer4.0.bn2.weight",
    "module.encoder.layer4.0.bn2.bias",
    "module.encoder.layer4.0.bn2.running_mean",
    "module.encoder.layer4.0.bn2.running_var",
    "module.encoder.layer4.0.bn2.num_batches_tracked",
    "module.encoder.layer4.0.conv3.weight",
    "module.encoder.layer4.0.bn3.weight",
    "module.encoder.layer4.0.bn3.bias",
    "module.encoder.layer4.0.bn3.running_mean",
    "module.encoder.layer4.0.bn3.running_var",
    "module.encoder.layer4.0.bn3.num_batches_tracked",
    "module.encoder.layer4.0.downsample.0.weight",
    "module.encoder.layer4.0.downsample.1.weight",
    "module.encoder.layer4.0.downsample.1.bias",
    "module.encoder.layer4.0.downsample.1.running_mean",
    "module.encoder.layer4.0.downsample.1.running_var",
    "module.encoder.layer4.0.downsample.1.num_batches_tracked",
    "module.encoder.layer4.1.conv1.weight",
    "module.encoder.layer4.1.bn1.weight",
    "module.encoder.layer4.1.bn1.bias",
    "module.encoder.layer4.1.bn1.running_mean",
    "module.encoder.layer4.1.bn1.running_var",
    "module.encoder.layer4.1.bn1.num_batches_tracked",
    "module.encoder.layer4.1.conv2.weight",
    "module.encoder.layer4.1.bn2.weight",
    "module.encoder.layer4.1.bn2.bias",
    "module.encoder.layer4.1.bn2.running_mean",
    "module.encoder.layer4.1.bn2.running_var",
    "module.encoder.layer4.1.bn2.num_batches_tracked",
    "module.encoder.layer4.1.conv3.weight",
    "module.encoder.layer4.1.bn3.weight",
    "module.encoder.layer4.1.bn3.bias",
    "module.encoder.layer4.1.bn3.running_mean",
    "module.encoder.layer4.1.bn3.running_var",
    "module.encoder.layer4.1.bn3.num_batches_tracked",
    "module.encoder.layer4.2.conv1.weight",
    "module.encoder.layer4.2.bn1.weight",
    "module.encoder.layer4.2.bn1.bias",
    "module.encoder.layer4.2.bn1.running_mean",
    "module.encoder.layer4.2.bn1.running_var",
    "module.encoder.layer4.2.bn1.num_batches_tracked",
    "module.encoder.layer4.2.conv2.weight",
    "module.encoder.layer4.2.bn2.weight",
    "module.encoder.layer4.2.bn2.bias",
    "module.encoder.layer4.2.bn2.running_mean",
    "module.encoder.layer4.2.bn2.running_var",
    "module.encoder.layer4.2.bn2.num_batches_tracked",
    "module.encoder.layer4.2.conv3.weight",
    "module.encoder.layer4.2.bn3.weight",
    "module.encoder.layer4.2.bn3.bias",
    "module.encoder.layer4.2.bn3.running_mean",
    "module.encoder.layer4.2.bn3.running_var",
    "module.encoder.layer4.2.bn3.num_batches_tracked",
    "module.projector.linear1.weight",
    "module.projector.linear1.bias",
    "module.projector.bn1.weight",
    "module.projector.bn1.bias",
    "module.projector.bn1.running_mean",
    "module.projector.bn1.running_var",
    "module.projector.bn1.num_batches_tracked",
    "module.projector.linear2.weight",
    "module.projector.linear2.bias",
    "module.encoder_k.conv1.weight",
    "module.encoder_k.bn1.weight",
    "module.encoder_k.bn1.bias",
    "module.encoder_k.bn1.running_mean",
    "module.encoder_k.bn1.running_var",
    "module.encoder_k.bn1.num_batches_tracked",
    "module.encoder_k.layer1.0.conv1.weight",
    "module.encoder_k.layer1.0.bn1.weight",
    "module.encoder_k.layer1.0.bn1.bias",
    "module.encoder_k.layer1.0.bn1.running_mean",
    "module.encoder_k.layer1.0.bn1.running_var",
    "module.encoder_k.layer1.0.bn1.num_batches_tracked",
    "module.encoder_k.layer1.0.conv2.weight",
    "module.encoder_k.layer1.0.bn2.weight",
    "module.encoder_k.layer1.0.bn2.bias",
    "module.encoder_k.layer1.0.bn2.running_mean",
    "module.encoder_k.layer1.0.bn2.running_var",
    "module.encoder_k.layer1.0.bn2.num_batches_tracked",
    "module.encoder_k.layer1.0.conv3.weight",
    "module.encoder_k.layer1.0.bn3.weight",
    "module.encoder_k.layer1.0.bn3.bias",
    "module.encoder_k.layer1.0.bn3.running_mean",
    "module.encoder_k.layer1.0.bn3.running_var",
    "module.encoder_k.layer1.0.bn3.num_batches_tracked",
    "module.encoder_k.layer1.0.downsample.0.weight",
    "module.encoder_k.layer1.0.downsample.1.weight",
    "module.encoder_k.layer1.0.downsample.1.bias",
    "module.encoder_k.layer1.0.downsample.1.running_mean",
    "module.encoder_k.layer1.0.downsample.1.running_var",
    "module.encoder_k.layer1.0.downsample.1.num_batches_tracked",
    "module.encoder_k.layer1.1.conv1.weight",
    "module.encoder_k.layer1.1.bn1.weight",
    "module.encoder_k.layer1.1.bn1.bias",
    "module.encoder_k.layer1.1.bn1.running_mean",
    "module.encoder_k.layer1.1.bn1.running_var",
    "module.encoder_k.layer1.1.bn1.num_batches_tracked",
    "module.encoder_k.layer1.1.conv2.weight",
    "module.encoder_k.layer1.1.bn2.weight",
    "module.encoder_k.layer1.1.bn2.bias",
    "module.encoder_k.layer1.1.bn2.running_mean",
    "module.encoder_k.layer1.1.bn2.running_var",
    "module.encoder_k.layer1.1.bn2.num_batches_tracked",
    "module.encoder_k.layer1.1.conv3.weight",
    "module.encoder_k.layer1.1.bn3.weight",
    "module.encoder_k.layer1.1.bn3.bias",
    "module.encoder_k.layer1.1.bn3.running_mean",
    "module.encoder_k.layer1.1.bn3.running_var",
    "module.encoder_k.layer1.1.bn3.num_batches_tracked",
    "module.encoder_k.layer1.2.conv1.weight",
    "module.encoder_k.layer1.2.bn1.weight",
    "module.encoder_k.layer1.2.bn1.bias",
    "module.encoder_k.layer1.2.bn1.running_mean",
    "module.encoder_k.layer1.2.bn1.running_var",
    "module.encoder_k.layer1.2.bn1.num_batches_tracked",
    "module.encoder_k.layer1.2.conv2.weight",
    "module.encoder_k.layer1.2.bn2.weight",
    "module.encoder_k.layer1.2.bn2.bias",
    "module.encoder_k.layer1.2.bn2.running_mean",
    "module.encoder_k.layer1.2.bn2.running_var",
    "module.encoder_k.layer1.2.bn2.num_batches_tracked",
    "module.encoder_k.layer1.2.conv3.weight",
    "module.encoder_k.layer1.2.bn3.weight",
    "module.encoder_k.layer1.2.bn3.bias",
    "module.encoder_k.layer1.2.bn3.running_mean",
    "module.encoder_k.layer1.2.bn3.running_var",
    "module.encoder_k.layer1.2.bn3.num_batches_tracked",
    "module.encoder_k.layer2.0.conv1.weight",
    "module.encoder_k.layer2.0.bn1.weight",
    "module.encoder_k.layer2.0.bn1.bias",
    "module.encoder_k.layer2.0.bn1.running_mean",
    "module.encoder_k.layer2.0.bn1.running_var",
    "module.encoder_k.layer2.0.bn1.num_batches_tracked",
    "module.encoder_k.layer2.0.conv2.weight",
    "module.encoder_k.layer2.0.bn2.weight",
    "module.encoder_k.layer2.0.bn2.bias",
    "module.encoder_k.layer2.0.bn2.running_mean",
    "module.encoder_k.layer2.0.bn2.running_var",
    "module.encoder_k.layer2.0.bn2.num_batches_tracked",
    "module.encoder_k.layer2.0.conv3.weight",
    "module.encoder_k.layer2.0.bn3.weight",
    "module.encoder_k.layer2.0.bn3.bias",
    "module.encoder_k.layer2.0.bn3.running_mean",
    "module.encoder_k.layer2.0.bn3.running_var",
    "module.encoder_k.layer2.0.bn3.num_batches_tracked",
    "module.encoder_k.layer2.0.downsample.0.weight",
    "module.encoder_k.layer2.0.downsample.1.weight",
    "module.encoder_k.layer2.0.downsample.1.bias",
    "module.encoder_k.layer2.0.downsample.1.running_mean",
    "module.encoder_k.layer2.0.downsample.1.running_var",
    "module.encoder_k.layer2.0.downsample.1.num_batches_tracked",
    "module.encoder_k.layer2.1.conv1.weight",
    "module.encoder_k.layer2.1.bn1.weight",
    "module.encoder_k.layer2.1.bn1.bias",
    "module.encoder_k.layer2.1.bn1.running_mean",
    "module.encoder_k.layer2.1.bn1.running_var",
    "module.encoder_k.layer2.1.bn1.num_batches_tracked",
    "module.encoder_k.layer2.1.conv2.weight",
    "module.encoder_k.layer2.1.bn2.weight",
    "module.encoder_k.layer2.1.bn2.bias",
    "module.encoder_k.layer2.1.bn2.running_mean",
    "module.encoder_k.layer2.1.bn2.running_var",
    "module.encoder_k.layer2.1.bn2.num_batches_tracked",
    "module.encoder_k.layer2.1.conv3.weight",
    "module.encoder_k.layer2.1.bn3.weight",
    "module.encoder_k.layer2.1.bn3.bias",
    "module.encoder_k.layer2.1.bn3.running_mean",
    "module.encoder_k.layer2.1.bn3.running_var",
    "module.encoder_k.layer2.1.bn3.num_batches_tracked",
    "module.encoder_k.layer2.2.conv1.weight",
    "module.encoder_k.layer2.2.bn1.weight",
    "module.encoder_k.layer2.2.bn1.bias",
    "module.encoder_k.layer2.2.bn1.running_mean",
    "module.encoder_k.layer2.2.bn1.running_var",
    "module.encoder_k.layer2.2.bn1.num_batches_tracked",
    "module.encoder_k.layer2.2.conv2.weight",
    "module.encoder_k.layer2.2.bn2.weight",
    "module.encoder_k.layer2.2.bn2.bias",
    "module.encoder_k.layer2.2.bn2.running_mean",
    "module.encoder_k.layer2.2.bn2.running_var",
    "module.encoder_k.layer2.2.bn2.num_batches_tracked",
    "module.encoder_k.layer2.2.conv3.weight",
    "module.encoder_k.layer2.2.bn3.weight",
    "module.encoder_k.layer2.2.bn3.bias",
    "module.encoder_k.layer2.2.bn3.running_mean",
    "module.encoder_k.layer2.2.bn3.running_var",
    "module.encoder_k.layer2.2.bn3.num_batches_tracked",
    "module.encoder_k.layer2.3.conv1.weight",
    "module.encoder_k.layer2.3.bn1.weight",
    "module.encoder_k.layer2.3.bn1.bias",
    "module.encoder_k.layer2.3.bn1.running_mean",
    "module.encoder_k.layer2.3.bn1.running_var",
    "module.encoder_k.layer2.3.bn1.num_batches_tracked",
    "module.encoder_k.layer2.3.conv2.weight",
    "module.encoder_k.layer2.3.bn2.weight",
    "module.encoder_k.layer2.3.bn2.bias",
    "module.encoder_k.layer2.3.bn2.running_mean",
    "module.encoder_k.layer2.3.bn2.running_var",
    "module.encoder_k.layer2.3.bn2.num_batches_tracked",
    "module.encoder_k.layer2.3.conv3.weight",
    "module.encoder_k.layer2.3.bn3.weight",
    "module.encoder_k.layer2.3.bn3.bias",
    "module.encoder_k.layer2.3.bn3.running_mean",
    "module.encoder_k.layer2.3.bn3.running_var",
    "module.encoder_k.layer2.3.bn3.num_batches_tracked",
    "module.encoder_k.layer3.0.conv1.weight",
    "module.encoder_k.layer3.0.bn1.weight",
    "module.encoder_k.layer3.0.bn1.bias",
    "module.encoder_k.layer3.0.bn1.running_mean",
    "module.encoder_k.layer3.0.bn1.running_var",
    "module.encoder_k.layer3.0.bn1.num_batches_tracked",
    "module.encoder_k.layer3.0.conv2.weight",
    "module.encoder_k.layer3.0.bn2.weight",
    "module.encoder_k.layer3.0.bn2.bias",
    "module.encoder_k.layer3.0.bn2.running_mean",
    "module.encoder_k.layer3.0.bn2.running_var",
    "module.encoder_k.layer3.0.bn2.num_batches_tracked",
    "module.encoder_k.layer3.0.conv3.weight",
    "module.encoder_k.layer3.0.bn3.weight",
    "module.encoder_k.layer3.0.bn3.bias",
    "module.encoder_k.layer3.0.bn3.running_mean",
    "module.encoder_k.layer3.0.bn3.running_var",
    "module.encoder_k.layer3.0.bn3.num_batches_tracked",
    "module.encoder_k.layer3.0.downsample.0.weight",
    "module.encoder_k.layer3.0.downsample.1.weight",
    "module.encoder_k.layer3.0.downsample.1.bias",
    "module.encoder_k.layer3.0.downsample.1.running_mean",
    "module.encoder_k.layer3.0.downsample.1.running_var",
    "module.encoder_k.layer3.0.downsample.1.num_batches_tracked",
    "module.encoder_k.layer3.1.conv1.weight",
    "module.encoder_k.layer3.1.bn1.weight",
    "module.encoder_k.layer3.1.bn1.bias",
    "module.encoder_k.layer3.1.bn1.running_mean",
    "module.encoder_k.layer3.1.bn1.running_var",
    "module.encoder_k.layer3.1.bn1.num_batches_tracked",
    "module.encoder_k.layer3.1.conv2.weight",
    "module.encoder_k.layer3.1.bn2.weight",
    "module.encoder_k.layer3.1.bn2.bias",
    "module.encoder_k.layer3.1.bn2.running_mean",
    "module.encoder_k.layer3.1.bn2.running_var",
    "module.encoder_k.layer3.1.bn2.num_batches_tracked",
    "module.encoder_k.layer3.1.conv3.weight",
    "module.encoder_k.layer3.1.bn3.weight",
    "module.encoder_k.layer3.1.bn3.bias",
    "module.encoder_k.layer3.1.bn3.running_mean",
    "module.encoder_k.layer3.1.bn3.running_var",
    "module.encoder_k.layer3.1.bn3.num_batches_tracked",
    "module.encoder_k.layer3.2.conv1.weight",
    "module.encoder_k.layer3.2.bn1.weight",
    "module.encoder_k.layer3.2.bn1.bias",
    "module.encoder_k.layer3.2.bn1.running_mean",
    "module.encoder_k.layer3.2.bn1.running_var",
    "module.encoder_k.layer3.2.bn1.num_batches_tracked",
    "module.encoder_k.layer3.2.conv2.weight",
    "module.encoder_k.layer3.2.bn2.weight",
    "module.encoder_k.layer3.2.bn2.bias",
    "module.encoder_k.layer3.2.bn2.running_mean",
    "module.encoder_k.layer3.2.bn2.running_var",
    "module.encoder_k.layer3.2.bn2.num_batches_tracked",
    "module.encoder_k.layer3.2.conv3.weight",
    "module.encoder_k.layer3.2.bn3.weight",
    "module.encoder_k.layer3.2.bn3.bias",
    "module.encoder_k.layer3.2.bn3.running_mean",
    "module.encoder_k.layer3.2.bn3.running_var",
    "module.encoder_k.layer3.2.bn3.num_batches_tracked",
    "module.encoder_k.layer3.3.conv1.weight",
    "module.encoder_k.layer3.3.bn1.weight",
    "module.encoder_k.layer3.3.bn1.bias",
    "module.encoder_k.layer3.3.bn1.running_mean",
    "module.encoder_k.layer3.3.bn1.running_var",
    "module.encoder_k.layer3.3.bn1.num_batches_tracked",
    "module.encoder_k.layer3.3.conv2.weight",
    "module.encoder_k.layer3.3.bn2.weight",
    "module.encoder_k.layer3.3.bn2.bias",
    "module.encoder_k.layer3.3.bn2.running_mean",
    "module.encoder_k.layer3.3.bn2.running_var",
    "module.encoder_k.layer3.3.bn2.num_batches_tracked",
    "module.encoder_k.layer3.3.conv3.weight",
    "module.encoder_k.layer3.3.bn3.weight",
    "module.encoder_k.layer3.3.bn3.bias",
    "module.encoder_k.layer3.3.bn3.running_mean",
    "module.encoder_k.layer3.3.bn3.running_var",
    "module.encoder_k.layer3.3.bn3.num_batches_tracked",
    "module.encoder_k.layer3.4.conv1.weight",
    "module.encoder_k.layer3.4.bn1.weight",
    "module.encoder_k.layer3.4.bn1.bias",
    "module.encoder_k.layer3.4.bn1.running_mean",
    "module.encoder_k.layer3.4.bn1.running_var",
    "module.encoder_k.layer3.4.bn1.num_batches_tracked",
    "module.encoder_k.layer3.4.conv2.weight",
    "module.encoder_k.layer3.4.bn2.weight",
    "module.encoder_k.layer3.4.bn2.bias",
    "module.encoder_k.layer3.4.bn2.running_mean",
    "module.encoder_k.layer3.4.bn2.running_var",
    "module.encoder_k.layer3.4.bn2.num_batches_tracked",
    "module.encoder_k.layer3.4.conv3.weight",
    "module.encoder_k.layer3.4.bn3.weight",
    "module.encoder_k.layer3.4.bn3.bias",
    "module.encoder_k.layer3.4.bn3.running_mean",
    "module.encoder_k.layer3.4.bn3.running_var",
    "module.encoder_k.layer3.4.bn3.num_batches_tracked",
    "module.encoder_k.layer3.5.conv1.weight",
    "module.encoder_k.layer3.5.bn1.weight",
    "module.encoder_k.layer3.5.bn1.bias",
    "module.encoder_k.layer3.5.bn1.running_mean",
    "module.encoder_k.layer3.5.bn1.running_var",
    "module.encoder_k.layer3.5.bn1.num_batches_tracked",
    "module.encoder_k.layer3.5.conv2.weight",
    "module.encoder_k.layer3.5.bn2.weight",
    "module.encoder_k.layer3.5.bn2.bias",
    "module.encoder_k.layer3.5.bn2.running_mean",
    "module.encoder_k.layer3.5.bn2.running_var",
    "module.encoder_k.layer3.5.bn2.num_batches_tracked",
    "module.encoder_k.layer3.5.conv3.weight",
    "module.encoder_k.layer3.5.bn3.weight",
    "module.encoder_k.layer3.5.bn3.bias",
    "module.encoder_k.layer3.5.bn3.running_mean",
    "module.encoder_k.layer3.5.bn3.running_var",
    "module.encoder_k.layer3.5.bn3.num_batches_tracked",
    "module.encoder_k.layer4.0.conv1.weight",
    "module.encoder_k.layer4.0.bn1.weight",
    "module.encoder_k.layer4.0.bn1.bias",
    "module.encoder_k.layer4.0.bn1.running_mean",
    "module.encoder_k.layer4.0.bn1.running_var",
    "module.encoder_k.layer4.0.bn1.num_batches_tracked",
    "module.encoder_k.layer4.0.conv2.weight",
    "module.encoder_k.layer4.0.bn2.weight",
    "module.encoder_k.layer4.0.bn2.bias",
    "module.encoder_k.layer4.0.bn2.running_mean",
    "module.encoder_k.layer4.0.bn2.running_var",
    "module.encoder_k.layer4.0.bn2.num_batches_tracked",
    "module.encoder_k.layer4.0.conv3.weight",
    "module.encoder_k.layer4.0.bn3.weight",
    "module.encoder_k.layer4.0.bn3.bias",
    "module.encoder_k.layer4.0.bn3.running_mean",
    "module.encoder_k.layer4.0.bn3.running_var",
    "module.encoder_k.layer4.0.bn3.num_batches_tracked",
    "module.encoder_k.layer4.0.downsample.0.weight",
    "module.encoder_k.layer4.0.downsample.1.weight",
    "module.encoder_k.layer4.0.downsample.1.bias",
    "module.encoder_k.layer4.0.downsample.1.running_mean",
    "module.encoder_k.layer4.0.downsample.1.running_var",
    "module.encoder_k.layer4.0.downsample.1.num_batches_tracked",
    "module.encoder_k.layer4.1.conv1.weight",
    "module.encoder_k.layer4.1.bn1.weight",
    "module.encoder_k.layer4.1.bn1.bias",
    "module.encoder_k.layer4.1.bn1.running_mean",
    "module.encoder_k.layer4.1.bn1.running_var",
    "module.encoder_k.layer4.1.bn1.num_batches_tracked",
    "module.encoder_k.layer4.1.conv2.weight",
    "module.encoder_k.layer4.1.bn2.weight",
    "module.encoder_k.layer4.1.bn2.bias",
    "module.encoder_k.layer4.1.bn2.running_mean",
    "module.encoder_k.layer4.1.bn2.running_var",
    "module.encoder_k.layer4.1.bn2.num_batches_tracked",
    "module.encoder_k.layer4.1.conv3.weight",
    "module.encoder_k.layer4.1.bn3.weight",
    "module.encoder_k.layer4.1.bn3.bias",
    "module.encoder_k.layer4.1.bn3.running_mean",
    "module.encoder_k.layer4.1.bn3.running_var",
    "module.encoder_k.layer4.1.bn3.num_batches_tracked",
    "module.encoder_k.layer4.2.conv1.weight",
    "module.encoder_k.layer4.2.bn1.weight",
    "module.encoder_k.layer4.2.bn1.bias",
    "module.encoder_k.layer4.2.bn1.running_mean",
    "module.encoder_k.layer4.2.bn1.running_var",
    "module.encoder_k.layer4.2.bn1.num_batches_tracked",
    "module.encoder_k.layer4.2.conv2.weight",
    "module.encoder_k.layer4.2.bn2.weight",
    "module.encoder_k.layer4.2.bn2.bias",
    "module.encoder_k.layer4.2.bn2.running_mean",
    "module.encoder_k.layer4.2.bn2.running_var",
    "module.encoder_k.layer4.2.bn2.num_batches_tracked",
    "module.encoder_k.layer4.2.conv3.weight",
    "module.encoder_k.layer4.2.bn3.weight",
    "module.encoder_k.layer4.2.bn3.bias",
    "module.encoder_k.layer4.2.bn3.running_mean",
    "module.encoder_k.layer4.2.bn3.running_var",
    "module.encoder_k.layer4.2.bn3.num_batches_tracked",
    "module.projector_k.linear1.weight",
    "module.projector_k.linear1.bias",
    "module.projector_k.bn1.weight",
    "module.projector_k.bn1.bias",
    "module.projector_k.bn1.running_mean",
    "module.projector_k.bn1.running_var",
    "module.projector_k.bn1.num_batches_tracked",
    "module.projector_k.linear2.weight",
    "module.projector_k.linear2.bias",
    "module.value_transform.weight",
    "module.value_transform.bias",
]
