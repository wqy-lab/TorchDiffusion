import argparse
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_time import UNet
from models.diffusion import Diffusion
from utils.run_dir import (
    create_run_dir,
    get_latest_run_dir,
    init_train_log,
    log_epoch,
    save_run_config,
    write_latest_pointer,
)
import config


def get_data_loader():
    if config.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),  # [0,1] -> [-1,1]
        ])
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            transform=transform,
            download=True,
        )
    elif config.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(config.image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
        ])
        dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            transform=transform,
            download=True,
        )
    else:
        raise ValueError(f'Unsupported dataset: {config.dataset}')

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def resolve_init_checkpoint(init_checkpoint):
    if init_checkpoint is None:
        return None
    if init_checkpoint == 'latest':
        run_dir = get_latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError(
                f'No previous run found under {config.checkpoint_base}'
            )
        return os.path.join(run_dir, 'best_model.pt')
    return init_checkpoint


def load_init_weights(model, init_checkpoint, device):
    ckpt_path = resolve_init_checkpoint(init_checkpoint)
    if ckpt_path is None:
        print('Initializing model with default PyTorch weights')
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Initial checkpoint not found: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(
        f'Loaded initial weights from {ckpt_path} '
        f'(epoch={ckpt.get("epoch", "?")}, loss={ckpt.get("loss", "?")})'
    )


def save_checkpoint(path, model, optimizer, epoch, loss, timestamp, run_dir):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'dataset': config.dataset,
        'timestamp': timestamp,
        'run_dir': run_dir,
    }, path)


def train(init_checkpoint=None):
    device = config.device
    run_dir, timestamp = create_run_dir()
    log_path = init_train_log(run_dir, timestamp)
    save_run_config(run_dir, timestamp, init_checkpoint=init_checkpoint)
    write_latest_pointer(run_dir)

    print(f'Using device: {device}')
    print(f'Dataset: {config.dataset}, image_size={config.image_size}, channels={config.in_channels}')
    print(f'Run directory: {run_dir}')

    train_loader = get_data_loader()

    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        num_groups=config.num_groups,
        time_dim=config.time_dim,
    ).to(device)
    load_init_weights(model, init_checkpoint, device)

    diffusion = Diffusion(
        T=config.T,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_loss = float('inf')
    epochs_without_improvement = 0
    final_epoch = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for step, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            t = torch.randint(0, config.T, (images.size(0),), device=device)
            noisy_images, noise = diffusion.q_sample(images, t, torch.randn_like(images))
            pred_noise = model(noisy_images, t)
            loss = nn.MSELoss()(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / (step + 1)
        final_epoch = epoch + 1
        print(f'[Epoch {final_epoch}/{config.epochs}] avg_loss={avg_loss:.6f}')

        improved = avg_loss < best_loss - config.early_stopping_min_delta
        log_epoch(log_path, final_epoch, avg_loss, improved)

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(run_dir, f'model_epoch_{epoch+1}.pt')
            save_checkpoint(
                ckpt_path, model, optimizer, epoch + 1, avg_loss, timestamp, run_dir
            )
            print(f'Checkpoint saved: {ckpt_path}')

        if improved:
            best_loss = avg_loss
            epochs_without_improvement = 0
            best_path = os.path.join(run_dir, 'best_model.pt')
            save_checkpoint(
                best_path, model, optimizer, epoch + 1, best_loss, timestamp, run_dir
            )
            print(f'Best model updated: loss={best_loss:.6f}')
        else:
            epochs_without_improvement += 1
            print(
                f'No improvement for {epochs_without_improvement}/'
                f'{config.early_stopping_patience} epoch(s)'
            )
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    f'Early stopping triggered at epoch {epoch+1}, '
                    f'best_loss={best_loss:.6f}'
                )
                break

    last_path = os.path.join(run_dir, 'last_model.pt')
    save_checkpoint(
        last_path, model, optimizer, final_epoch, avg_loss, timestamp, run_dir
    )
    print(f'Last model saved: {last_path}')
    print('Training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train diffusion model')
    parser.add_argument(
        '--init',
        type=str,
        default=None,
        help='initial checkpoint path, or "latest" for the newest best_model.pt',
    )
    args = parser.parse_args()
    init_checkpoint = args.init if args.init is not None else config.init_checkpoint
    train(init_checkpoint=init_checkpoint)
