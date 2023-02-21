import argparse

from runner import train, eval, ddp


def parse():
    parser = argparse.ArgumentParser("Diffusion Model")
    parser.add_argument("--state", type=str, choices=["train", "eval", "ddp"], default="train",
        help="train or eval the network")
    
    # training parameters
    parser.add_argument("--epoch", type=int, default=1000, help="number of epoches")
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
    parser.add_argument("--eta", type=float, default=1, help="coefficient to adjust sample process")
    parser.add_argument("--timesteps", type=int, default=1000, help="time step in sample process")
    parser.add_argument("--time_schedule", type=str, default="uniform", choices=["uniform", "quad"], help="skip type")

    # optimizer configuration
    parser.add_argument("--multiplier", type=float, default=2.0, help="learning rate multiplier during warm up state")

    # training configuration
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip during training")

    # directory configuration
    parser.add_argument("--test_time", type=int, default=50, help="interval to save model")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="path to load/save model")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="specify which checkpoint to load")
    parser.add_argument("--sample_dir", type=str, default="./samples", help="path to save sampled images")
    parser.add_argument("--sample_noise_name", type=str, default="NoisyImgs.png", help="save sampled gaussian noises")
    parser.add_argument("--sample_image_name", type=str, default="SampleImgs.png", help="save sampled images")

    # demonstrate configuration
    parser.add_argument("--nrow", type=int, default=8, help="number of rows to demonstrate")

    # datasets
    parser.add_argument("--img_size", type=int, default=32, help="image size")
    parser.add_argument("--dataset", type=str, default="cifar10", help="datasets for training")

    # device
    parser.add_argument("--device", type=str, default="cuda:1", help="use to cpu/cuda")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    if args.state == "train":
        train(args)
    elif args.state == 'eval':
        eval(args)
    elif args.state == 'ddp':
        ddp(args)
