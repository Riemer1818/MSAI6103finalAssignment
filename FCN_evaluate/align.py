import torch
import numpy as np
import os
import torch.nn as nn


def align_model_state_keys(state_dict, mapping):
    new_state_dict = {}
    for key in state_dict:
        if key in mapping:
            new_state_dict[mapping[key]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict

# Define the mapping from unexpected keys to missing keys
key_mapping = {
    "features.0.weight": "conv1_1.weight",
    "features.0.bias": "conv1_1.bias",
    "features.2.weight": "conv1_2.weight",
    "features.2.bias": "conv1_2.bias",
    "features.5.weight": "conv2_1.weight",
    "features.5.bias": "conv2_1.bias",
    "features.7.weight": "conv2_2.weight",
    "features.7.bias": "conv2_2.bias",
    "features.10.weight": "conv3_1.weight",
    "features.10.bias": "conv3_1.bias",
    "features.12.weight": "conv3_2.weight",
    "features.12.bias": "conv3_2.bias",
    "features.14.weight": "conv3_3.weight",
    "features.14.bias": "conv3_3.bias",
    "features.17.weight": "conv4_1.weight",
    "features.17.bias": "conv4_1.bias",
    "features.19.weight": "conv4_2.weight",
    "features.19.bias": "conv4_2.bias",
    "features.21.weight": "conv4_3.weight",
    "features.21.bias": "conv4_3.bias",
    "features.24.weight": "conv5_1.weight",
    "features.24.bias": "conv5_1.bias",
    "features.26.weight": "conv5_2.weight",
    "features.26.bias": "conv5_2.bias",
    "features.28.weight": "conv5_3.weight",
    "features.28.bias": "conv5_3.bias",
    "classifier.0.weight": "fc6.weight",
    "classifier.0.bias": "fc6.bias",
    "classifier.3.weight": "fc7.weight",
    "classifier.3.bias": "fc7.bias",
    "classifier.6.weight": "score_fr.weight",
    "classifier.6.bias": "score_fr.bias",
}

# Path to your model's state dictionary
model_path = './vgg16_caffe.pth'

# Load the original state dictionary
original_state_dict = torch.load(model_path)

# Align the keys
aligned_state_dict = align_model_state_keys(original_state_dict, key_mapping)
torch.save(aligned_state_dict, './vgg16_caffe2.pth')
# Assuming you have a model defined (e.g., model = YourModelClass())
# You can now load the aligned state dictionary into your model
# model.load_state_dict(aligned_state_dict)

# If you want to save the aligned state dictionary for later use
# torch.save(aligned_state_dict, 'path_to_save_aligned_model.pth')