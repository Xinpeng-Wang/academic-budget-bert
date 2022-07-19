import os
import sys
import collections
import torch
from multiprocessing import Pool
import yaml
from pathlib import Path
from datetime import datetime

finetune_config = {
    "--model_name_or_path": "training-out/teacher_test",
  "--task_name": "rte",
  "--max_seq_length": 128,
  "--output_dir": "/tmp/finetuning",
  "--overwrite_output_dir": None,
  "--do_train": None, 
  "--do_eval": None,
  "--evaluation_strategy": "steps",
  "--per_device_train_batch_size": 32, 
  "--gradient_accumulation_steps": 1 ,
  "--per_device_eval_batch_size": 32,
  "--learning_rate": 5e-5,
  "--weight_decay": 0.01,
  "--eval_steps": 100, 
  "--evaluation_strategy": "steps",
  "--max_grad_norm": 1.0,
  "--num_train_epochs": 20,
  "--run_name": "test",
  "--warmup_ratio": 0.06
}



def train(device, config):

    cmd = f'CUDA_VISIBLE_DEVICES={device} python run_glue.py ' 
    options = []
    for k, v in config.items():   
        if v is not None:
            options += [ f'{k} {v}']  
        else: options += [ f'{k}']



    cmd += ' '.join(options)

    os.system(cmd)


def experiment_start(task, device, batch_size, lr):
    config = finetune_config
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%R")
    save_dir = f"runs/{current_time}"


    config['--learning_rate'] = lr
    config['--output_dir'] = save_dir
    config['--task_name'] = task
    config['--per_device_train_batch_size'] = batch_size
    config['--run_name'] = f'{task}_{lr}lr_{batch_size}bz'

    train(device, config)



num_workers = 8


for task in ['rte', 'cola', 'mrpc', 'qnli', 'sst2', 'mnli', 'qqp']:
    args = []
    idx = 0
    for lr in [1e-5, 3e-5, 5e-5, 8e-5]:
        for bz in [16, 32]:
            args.append((task, idx, bz, lr))
    

    writer_workers = Pool(num_workers)
    writer_workers.starmap(experiment_start, args)



