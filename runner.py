import torch
from torch import optim
from torchvision.utils import save_image
import os
from tqdm import tqdm
import glob

from model import UNet
from diffusion import DiffusionTrainer, DiffusionSampler
from loader import load_dataset, save_dataset
from scheduler import WarmUpScheduler
from utils import inverse_data_transform

# Distributed Data Parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


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
        sampler = DiffusionSampler(net, args.beta_1, args.beta_T, args.T,
                                   args.eta, args.timesteps, args.time_schedule).to(device)

        if not os.path.exists(args.sample_dir):
            os.makedirs(args.sample_dir)

        # sample from gaussian distribution
        noisyImage = torch.randn(size=[args.batch_size, 3, 32, 32], device=device)
        noise = inverse_data_transform(args, noisyImage)
        save_image(noise, os.path.join(args.sample_dir, args.sample_noise_name), nrow=args.nrow)
        image = sampler(noisyImage)
        image = inverse_data_transform(args, image)
        save_image(image, os.path.join(args.sample_dir, args.sample_image_name), nrow=args.nrow)

        if args.fid:
            # generate images for eval
            gene_dir = os.path.join(args.sample_dir, "gen_" + args.dataset)
            if not os.path.exists(gene_dir):
                os.makedirs(gene_dir)
            
            total = 50000
            img_id = len(glob.glob(gene_dir))
            round = (total - img_id) // args.batch_size
            for i in range(round):
                noise = torch.randn(size=[args.batch_size, 3, 32, 32], device=device)
                image = sampler(noise)
                image = inverse_data_transform(args, image)
                for j in range(args.batch_size):
                    save_image(image[j], os.path.join(gene_dir, f"{img_id}.png"))
                    img_id += 1
                
                print(f"Process {i}/{round}")


# distributed data parallel setup
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


#distributed data parallel cleanup
def cleanup():
    dist.destroy_process_group()


# distributed data parallel training
def ddp_training(rank, args, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    # dataset
    dataset, dataloader = load_dataset(args)

    # model
    net = UNet(T=args.T, ch=args.channel, ch_mult=args.channel_mult, attn=args.attn,
                    num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(rank)
    ddp_net = DDP(net, device_ids=[rank])

    CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, "ckpt_ddp_0_.pt")
    if rank == 0:
        torch.save(ddp_net.state_dict(), CHECKPOINT_PATH)

    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))
    
    # optimizer
    optimizer = torch.optim.AdamW(
        ddp_net.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = WarmUpScheduler(
        optimizer=optimizer, multiplier=args.multiplier, warm_epoch=args.epoch // 10, after_scheduler=cosineScheduler)

    # trainer
    trainer = DiffusionTrainer(ddp_net, args.beta_1, args.beta_T, args.T).to(rank)

    # start training
    for e in range(args.epoch):
        if rank == 0:
            loader = tqdm(dataloader, dynamic_ncols=True)
        else:
            loader = dataloader
        for images, labels in loader:
            # train
            optimizer.zero_grad()
            x_0 = images.to(rank)
            loss = trainer(x_0).sum() / 1000.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_net.parameters(), args.grad_clip)
            optimizer.step()
            if rank == 0:
                loader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "img shape": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if (e + 1) % args.test_time == 0 and rank == 0:
            torch.save(ddp_net.state_dict(), os.path.join(args.checkpoint_dir, "ckpt_ddp_" + str(e) + "_.pt"))

    cleanup()


# distributed data parallel caller
def ddp(args):
    n_gpus = torch.cuda.device_count() - 1
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(ddp_training,
             args=(args, world_size, ),
             nprocs=world_size,
             join=True)
