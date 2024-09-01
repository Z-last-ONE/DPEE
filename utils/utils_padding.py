import torch


def pad_2d(data, new_data):
    for j, x in enumerate(data):
        new_data[j, :x.shape[0], :x.shape[1]] = x
    return new_data


def pad_3d(data, new_data):
    for j, x in enumerate(data):
        new_data[j, :x.shape[0], :x.shape[1], :] = x
    return new_data


def pad_4d(data, new_data):
    for j, x in enumerate(data):
        new_data[j, :, :x.shape[1], :x.shape[2], :] = x
    return new_data
