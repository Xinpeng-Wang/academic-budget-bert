import os
import sys
import collections
import torch
from multiprocessing import Pool
import yaml
from pathlib import Path
from datetime import datetime


finetune_config = {
"--model_name_or_path": "~/code/academic-budget-bert/training-out/general_distill_6layer_60h_continue-experiment/experiment/epoch1000000_step301541",
# "~/code/academic-budget-bert/training-out/general_distill_6layer_30h_from9layerTA-experiment/experiment/epoch1000000_step179332",
# "~/code/academic-budget-bert/training-out/general_distill_9layer_36h-experiment/experiment/epoch1000000_step141206",
# "~/code/academic-budget-bert/training-out/general_distill_6layer_30h-experiment/experiment/epoch1000000_step150619",
# 
# 
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
  "--num_train_epochs": 30,
  "--run_name": "test",
  "--warmup_ratio": 0.06,
  "--log_group":  "general_distill_6layer_60h_continue-experiment_epoch1000000_step301541"
#   "general_distill_6layer_30h_from9layerTA-experiment_epoch1000000_step179332"
#   "general_distill_9layer_36h-experiment_epoch1000000_step141206"
#   "general_distill_6layer_30h-experiment_epoch1000000_step150619"

}

os.environ["WANDB_DISABLED"] = "true"

def train(device, config):

    cmd = f'CUDA_VISIBLE_DEVICES=0 python run_glue.py ' 
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
    save_dir = f"runs/task"


    config['--learning_rate'] = lr
    config['--output_dir'] = save_dir
    config['--task_name'] = task
    config['--per_device_train_batch_size'] = batch_size

    config['--run_name'] = f'{task}_{lr}lr_{batch_size}bz_20epoch'

    train(device, config)




for task in ['rte', 'cola', 'mrpc', 'qnli', 'sst2', 'mnli', 'qqp']:
    args = []
    idx = 0
    for lr in [1e-5, 3e-5, 5e-5, 8e-5]:
        for bz in [16, 32]:
            args.append((task, idx, bz, lr))
    
    pass








