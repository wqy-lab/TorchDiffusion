import argparse
import math
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models.diffusion import Diffusion
from models.unet_time import UNet
from utils.run_dir import resolve_checkpoint
import config


def load_model(ckpt_path, device):
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        num_groups=config.num_groups,
        time_dim=config.time_dim,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


def upscale_for_export(images, scale):
    if scale <= 1:
        return images
    return F.interpolate(images, scale_factor=scale, mode='nearest')


def sample(num_images=16, run_name=None, ckpt=None, output_path=None, scale=None, padding=None):
    device = config.device
    ckpt_path, run_dir = resolve_checkpoint(run_name=run_name, ckpt=ckpt)

    if output_path is None:
        output_path = os.path.join(run_dir, 'samples.png')

    print(f'Using device: {device}')
    print(f'Dataset: {config.dataset}')
    print(f'Run: {run_dir}')
    print(f'Checkpoint: {ckpt_path}')

    model, ckpt_meta = load_model(ckpt_path, device)
    diffusion = Diffusion(
        T=config.T,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )

    print(
        f'Loaded epoch={ckpt_meta.get("epoch", "?")}, '
        f'loss={ckpt_meta.get("loss", "?")}'
    )
    print(f'Generating {num_images} images...')

    images = diffusion.sample(
        model,
        batch_size=num_images,
        image_size=config.image_size,
        channels=config.in_channels,
    )

    images = (images + 1) / 2  # [-1,1] -> [0,1]

    scale = config.sample_scale if scale is None else scale
    padding = config.sample_padding if padding is None else padding
    export_images = upscale_for_export(images, scale)
    nrow = max(1, int(math.sqrt(num_images)))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_image(export_images, output_path, nrow=nrow, padding=padding)
    export_size = config.image_size * scale
    print(
        f'Saved to {output_path} '
        f'({export_size}x{export_size} per image, scale={scale})'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample images from a trained diffusion model')
    parser.add_argument('-n', '--num-images', type=int, default=16, help='number of images to generate')
    parser.add_argument(
        '-r', '--run',
        type=str,
        default=None,
        help='training run folder name, or latest (default from config.sample_run)',
    )
    parser.add_argument(
        '-c', '--ckpt',
        type=str,
        default=None,
        help='best, last, 30, epoch:30, or full path (default from config.sample_ckpt)',
    )
    parser.add_argument('-o', '--output', type=str, default=None, help='output image path')
    parser.add_argument('-s', '--scale', type=int, default=None, help='export upscale factor')
    parser.add_argument('--padding', type=int, default=None, help='padding between images in grid')
    args = parser.parse_args()
    sample(
        num_images=args.num_images,
        run_name=args.run,
        ckpt=args.ckpt,
        output_path=args.output,
        scale=args.scale,
        padding=args.padding,
    )
