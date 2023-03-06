import torch
from torch import optim
from torchvision.utils import save_image
import os
from tqdm import tqdm
import glob

from tensorboardX import SummaryWriter

from model import UNet
from diffusion import DiffusionTrainer, DiffusionSampler
from loader import load_dataset
from scheduler import WarmUpScheduler
from utils import inverse_data_transform


# training diffusion model
def train(args):
    # device
    device = torch.device(args.device)
    # dataset
    train_data, train_loader = load_dataset(args)
    print(f"train {len(train_data)}")

    # model
    net = UNet(T=args.T, ch=args.channel, ch_mult=args.channel_mult, attn=args.attn,
                    num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(device)
    
    # ckpt
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.ckpt_name is not None:
        net.load_state_dict(torch.load(f"{args.ckpt_dir}/{args.ckpt_name}", map_location=device))

    # optimizer
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = WarmUpScheduler(
        optimizer=optimizer, multiplier=args.multiplier, warm_epoch=args.epoch // 10, after_scheduler=cosineScheduler)
    
    # trainer
    trainer = DiffusionTrainer(net, args.beta_1, args.beta_T, args.T).to(device)

    # summary writer
    dirs = glob.glob(f"{args.log_dir}/*")
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(f"{args.log_dir}/{len(dirs)}")

    # record
    last_epoch = -1
    global_step = 0

    # start training
    for epoch in range(last_epoch+1, args.epoch):
        with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
                summary_writer.add_scalars("loss", dict(train_loss=loss), global_step=global_step)
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss": loss.item(),
                    "img shape": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if (epoch + 1) % args.test_time == 0:
            torch.save(net.state_dict(), f"{args.ckpt_dir}/ckpt_{epoch}_.pt")


# sample from pre-trained model
def eval(args):
    # model
    with torch.no_grad():
        # device
        device = torch.device(args.device)
        # model
        net = UNet(T=args.T, ch=args.channel, ch_mult=args.channel_mult, attn=args.attn,
                        num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(device)

        # ckpt
        ckpt = torch.load(f"{args.ckpt_dir}/{args.ckpt_name}", map_location=device)
        net.load_state_dict(ckpt)
        print("model load weight done.")

        net.eval()
        sampler = DiffusionSampler(net, args.beta_1, args.beta_T, args.T,
                                   args.eta, args.timesteps, args.time_schedule).to(device)

        os.makedirs(args.sample_dir, exist_ok=True)

        # sample from gaussian distribution
        noisyImage = torch.randn(size=[args.batch_size, 3, 32, 32], device=device)
        noise = inverse_data_transform(args, noisyImage)
        save_image(noise, f"{args.sample_dir}/noise.png", nrow=args.nrow)
        image = sampler(noisyImage)
        image = inverse_data_transform(args, image)
        save_image(image, f"{args.sample_dir}/image.png", nrow=args.nrow)

        if args.fid:
            # generate images for eval
            export_dir = f"{args.sample_dir}/{args.dataset}"
            os.makedirs(export_dir, exist_ok=True)
            
            total = 2000
            img_id = len(glob.glob(export_dir))
            round = (total - img_id) // args.batch_size
            for i in tqdm(range(round)):
                noise = torch.randn(size=[args.batch_size, 3, 32, 32], device=device)
                image = sampler(noise)
                image = inverse_data_transform(args, image)
                for j in range(args.batch_size):
                    save_image(image[j], f"{export_dir}/{img_id}.png")
                    img_id += 1
