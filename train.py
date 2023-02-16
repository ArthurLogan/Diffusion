import argparse
import torch
from torch import optim
from torchvision.utils import save_image
import os
from tqdm import tqdm

from model import UNet
from diffusion import DiffusionTrainer, DiffusionSampler
from loader import load_dataset
from scheduler import WarmUpScheduler

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
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="use to cpu/cuda")

    args = parser.parse_args()
    return args


# training diffusion model
def train(args):
    # device
    device = torch.device(args.device)
    # dataset
    dataset, dataloader = load_dataset(args)

    # model
    net = UNet(T=args.T, ch=args.channel, ch_mult=args.channel_mult, attn=args.attn,
                    num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(device)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if args.checkpoint_name is not None:
        net.load_state_dict(torch.load(os.path.join(
            args.checkpoint_dir, args.checkpoint_name), map_location=device))

    # optimizer
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = WarmUpScheduler(
        optimizer=optimizer, multiplier=args.multiplier, warm_epoch=args.epoch // 10, after_scheduler=cosineScheduler)
    
    # trainer
    trainer = DiffusionTrainer(net, args.beta_1, args.beta_T, args.T).to(device)

    # start training
    for e in range(args.epoch):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "img shape": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if (e + 1) % args.test_time == 0:
            torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "ckpt_" + str(e) + "_.pt"))


# sample from pre-trained model
def eval(args):
    # model
    with torch.no_grad():
        # device
        device = torch.device(args.device)
        # model
        net = UNet(T=args.T, ch=args.channel, ch_mult=args.channel_mult, attn=args.attn,
                    num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(device)
        ckpt = torch.load(os.path.join(
            args.checkpoint_dir, args.checkpoint_name), map_location=device)
        net.load_state_dict(ckpt)
        print("model load weight done.")

        net.eval()
        sampler = DiffusionSampler(net, args.beta_1, args.beta_T, args.T).to(device)

        if not os.path.exists(args.sample_dir):
            os.makedirs(args.sample_dir)

        # sample from gaussian distribution
        noisyImage = torch.randn(size=[args.batch_size, 3, 32, 32], device=device)
        noise = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(noise, os.path.join(args.sample_dir, args.sample_noise_name), nrow=args.nrow)
        image = sampler(noisyImage)
        image = image * 0.5 + 0.5
        save_image(image, os.path.join(args.sample_dir, args.sample_image_name), nrow=args.nrow)


if __name__ == "__main__":
    args = parse()
    if args.state == "train":
        train(args)
    elif args.state == 'eval':
        eval(args)
