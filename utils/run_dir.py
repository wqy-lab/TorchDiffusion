import json
import os
from datetime import datetime

import config


def get_config_dict():
    return {
        'dataset': config.dataset,
        'T': config.T,
        'beta_start': config.beta_start,
        'beta_end': config.beta_end,
        'batch_size': config.batch_size,
        'lr': config.lr,
        'epochs': config.epochs,
        'early_stopping_patience': config.early_stopping_patience,
        'early_stopping_min_delta': config.early_stopping_min_delta,
        'image_size': config.image_size,
        'in_channels': config.in_channels,
        'out_channels': config.out_channels,
        'base_channels': config.base_channels,
        'channel_mults': config.channel_mults,
        'time_dim': config.time_dim,
        'num_groups': config.num_groups,
        'device': str(config.device),
    }


def create_run_dir():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    run_dir = os.path.join(config.checkpoint_base, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp


def save_run_config(run_dir, timestamp, init_checkpoint=None):
    config_path = os.path.join(run_dir, 'config.json')
    payload = {
        'timestamp': timestamp,
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'init_checkpoint': init_checkpoint,
        **get_config_dict(),
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def init_train_log(run_dir, timestamp):
    log_path = os.path.join(run_dir, 'train.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'# run={timestamp} dataset={config.dataset}\n')
        f.write('epoch\tavg_loss\tis_best\n')
    return log_path


def log_epoch(log_path, epoch, avg_loss, is_best):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'{epoch}\t{avg_loss:.6f}\t{int(is_best)}\n')


def write_latest_pointer(run_dir):
    pointer_path = os.path.join(config.checkpoint_base, 'latest.txt')
    with open(pointer_path, 'w', encoding='utf-8') as f:
        f.write(os.path.basename(run_dir))


def get_latest_run_dir():
    pointer_path = os.path.join(config.checkpoint_base, 'latest.txt')
    if os.path.exists(pointer_path):
        with open(pointer_path, encoding='utf-8') as f:
            run_name = f.read().strip()
        run_dir = os.path.join(config.checkpoint_base, run_name)
        if os.path.isdir(run_dir):
            return run_dir

    if not os.path.isdir(config.checkpoint_base):
        return None

    runs = [
        os.path.join(config.checkpoint_base, name)
        for name in os.listdir(config.checkpoint_base)
        if os.path.isdir(os.path.join(config.checkpoint_base, name))
    ]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def resolve_run_dir(run_name=None):
    if run_name is None or run_name == 'latest':
        run_dir = get_latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError(
                f'No training runs found under {config.checkpoint_base}'
            )
        return run_dir
    run_dir = os.path.join(config.checkpoint_base, run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f'Training run not found: {run_dir}')
    return run_dir


def resolve_checkpoint(run_name=None, ckpt=None):
    run_name = run_name or config.sample_run
    ckpt = ckpt or config.sample_ckpt
    run_dir = resolve_run_dir(run_name)

    if os.path.isfile(ckpt):
        return ckpt, run_dir

    ckpt_key = str(ckpt).lower()
    if ckpt_key in ('best', 'best_model', 'best_model.pt'):
        ckpt_path = os.path.join(run_dir, 'best_model.pt')
    elif ckpt_key in ('last', 'last_model', 'last_model.pt'):
        ckpt_path = os.path.join(run_dir, 'last_model.pt')
    elif ckpt_key.startswith('epoch:'):
        ckpt_path = os.path.join(run_dir, f'model_epoch_{ckpt_key.split(":", 1)[1]}.pt')
    elif ckpt_key.isdigit():
        ckpt_path = os.path.join(run_dir, f'model_epoch_{ckpt_key}.pt')
    else:
        ckpt_path = os.path.join(run_dir, ckpt_key)
        if not ckpt_path.endswith('.pt'):
            ckpt_path += '.pt'

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    return ckpt_path, run_dir
