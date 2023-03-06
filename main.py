import argparse
import json

from runner import train, eval
from utils import Dict



def parse():
    parser = argparse.ArgumentParser("Diffusion Model")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="train or eval the network")
    parser.add_argument("--config", type=str, default="config/train.json", help="path to configuration")
    parser.add_argument("--fid", action="store_true", help="if to calculate fid metric during sampling")
    parser.add_argument("--ckpt_name", type=str, default=None, help="specify which checkpoint to load")
    parser.add_argument("--device", type=str, default="cuda:1", help="use to cpu/cuda")
    args = parser.parse_args()

    with open(args.config) as file:
        config = Dict(json.load(file))
    config.update(vars(args))

    return config


if __name__ == "__main__":
    args = parse()
    if args.mode == "train":
        train(args)
    elif args.mode == 'eval':
        eval(args)
    else:
        raise Exception(f"undefined mode {args.mode}")
