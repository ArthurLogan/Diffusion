import argparse
from loader import load_dataset

def parse():
    parser = argparse.ArgumentParser("Diffusion Model")
    parser.add_argument("--state", type=str, choices=["train", "eval"], default="train",
        help="train or eval the network")
    
    # training parameters
    parser.add_argument("--epoch", type=int, default=200, help="number of epoches")
    parser.add_argument("--batch_size", type=int, default=80, help="number of batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    # unet parameters
    parser.add_argument("--channel", type=int, default=128, help="channel of first layer output")
    parser.add_argument("--channel_mult", type=list, default=[1, 2, 3, 4], help="channel multiplier during unet")
    parser.add_argument("--attn", type=list, default=[2], help="attention layer in layer")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="number of residual blocks in downsamples")
    parser.add_argument("--dropout", type=float, default=0.15, help="dropout rate")

    # diffusion parameters
    parser.add_argument("--T", type=int, default=1000, help="number of steps in reversed process")
    parser.add_argument("--beta_1", type=float, default=1e-4, help="forward process variance")
    parser.add_argument("--beta_T", type=float, default=0.02, help="forward process variance")

    # training configuration
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/", help="path to load/save model")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="specify which checkpoint to load")
    parser.add_argument("--sampled_dir", type=str, default="./samples/", help="path to save sampled images")

    # datasets
    parser.add_argument("--img_size", type=int, default=32, help="image size")
    parser.add_argument("--dataset", type=str, default="cifar10", help="datasets for training")

    args = parser.parse_args()
    return args


def train(args):
    dataset, dataloader = load_dataset(args)
    


if __name__ == "__main__":
    args = parse()
    if args.state == "train":
        train(args)
