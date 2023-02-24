import torch


# convert [-1, 1] -> [0, 1] for save
def inverse_data_transform(args, data):
    data = data * 0.5 + 0.5
    return torch.clamp(data, 0, 1)
