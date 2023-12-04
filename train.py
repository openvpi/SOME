import importlib
import logging
import os
import pathlib
import sys

import click
import lightning.pytorch as pl
import torch.utils.data
import yaml
from lightning.pytorch.loggers import TensorBoardLogger

import training.base_task
from utils.config_utils import read_full_config, print_config
from utils.training_utils import (
    DsModelCheckpoint, DsTQDMProgressBar,
    get_latest_checkpoint_path, get_strategy
)

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


@click.command(help='Train a SOME model')
@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
@click.option('--exp_name', required=True, metavar='EXP', help='Name of the experiment')
@click.option('--work_dir', required=False, metavar='DIR', help='Directory to save the experiment')
def train(config, exp_name, work_dir):
    config = pathlib.Path(config)
    config = read_full_config(config)
    print_config(config)
    if work_dir is None:
        work_dir = pathlib.Path(__file__).parent / 'experiments'
    else:
        work_dir = pathlib.Path(work_dir)
    work_dir = work_dir / exp_name
    assert not work_dir.exists() or work_dir.is_dir(), f'Path \'{work_dir}\' is not a directory.'
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / 'config.yaml', 'w', encoding='utf8') as f:
        yaml.safe_dump(config, f)
    config.update({'work_dir': str(work_dir)})

    if not config['nccl_p2p']:
        print("Disabling NCCL P2P")
        os.environ['NCCL_P2P_DISABLE'] = '1'

    pl.seed_everything(config['seed'], workers=True)
    assert config['task_cls'] != ''
    pkg = ".".join(config["task_cls"].split(".")[:-1])
    cls_name = config["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    assert issubclass(task_cls, training.BaseTask), f'Task class {task_cls} is not a subclass of {training.BaseTask}.'

    task = task_cls(config=config)

    # work_dir = pathlib.Path(config['work_dir'])
    trainer = pl.Trainer(
        accelerator=config['pl_trainer_accelerator'],
        devices=config['pl_trainer_devices'],
        num_nodes=config['pl_trainer_num_nodes'],
        strategy=get_strategy(config['pl_trainer_strategy']),
        precision=config['pl_trainer_precision'],
        callbacks=[
            DsModelCheckpoint(
                dirpath=work_dir,
                filename='model_ckpt_steps_{step}',
                auto_insert_metric_name=False,
                monitor='step',
                mode='max',
                save_last=False,
                # every_n_train_steps=config['val_check_interval'],
                save_top_k=config['num_ckpt_keep'],
                permanent_ckpt_start=config['permanent_ckpt_start'],
                permanent_ckpt_interval=config['permanent_ckpt_interval'],
                verbose=True
            ),
            # LearningRateMonitor(logging_interval='step'),
            DsTQDMProgressBar(),
        ],
        logger=TensorBoardLogger(
            save_dir=str(work_dir),
            name='lightning_logs',
            version='lastest'
        ),
        gradient_clip_val=config['clip_grad_norm'],
        val_check_interval=config['val_check_interval'] * config['accumulate_grad_batches'],
        # so this is global_steps
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        max_steps=config['max_updates'],
        use_distributed_sampler=False,
        num_sanity_val_steps=config['num_sanity_val_steps'],
        accumulate_grad_batches=config['accumulate_grad_batches']
    )
    trainer.fit(task, ckpt_path=get_latest_checkpoint_path(work_dir))


os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision


if __name__ == '__main__':
    train()
