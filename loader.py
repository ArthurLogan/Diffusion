from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def load_dataset(args):
    """dataloader"""
    if args.dataset == "cifar10":
        return load_cifar10(args.batch_size)
    else:
        raise ValueError("undefined datasets")


def load_cifar10(batch_size: int):
    """dataloader for cifar10 datasets"""
    dataset = CIFAR10(
        root='./datasets', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True
    )
    return dataset, dataloader
