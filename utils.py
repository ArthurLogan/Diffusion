import torch


class Dict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]


# convert [-1, 1] -> [0, 1] for save
def inverse_data_transform(args, data):
    data = data * 0.5 + 0.5
    return torch.clamp(data, 0, 1)
