import os
import sys
import collections
import torch
from multiprocessing import Pool
import yaml
from pathlib import Path
from datetime import datetime


finetune_config = {
"--model_name_or_path":  "~/code/academic-budget-bert/training-out/general_distill_6layer_30h-experiment/experiment/epoch1000000_step150619",
# "~/code/academic-budget-bert/training-out/general_distill_6layer_60h_continue-experiment/experiment/epoch1000000_step301541",
# "~/code/academic-budget-bert/training-out/general_distill_6layer_30h_from9layerTA-experiment/experiment/epoch1000000_step179332",

# 
# "~/code/academic-budget-bert/models/teachers/bert-base-uncased",
# "~/code/academic-budget-bert/training-out/general_distill_9layer_36h-experiment/experiment/epoch1000000_step141206",
# 
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
  "--eval_steps": 500, 
  "--evaluation_strategy": "steps",
  "--max_grad_norm": 1.0,
  "--num_train_epochs": 10,
  "--run_name": "test",
  "--warmup_ratio": 0.06,
  "--log_group":   "general_distill_6layer_30h-experiment_epoch1000000_step150619"
#   "general_distill_6layer_60h_continue-experiment_epoch1000000_step301541"
#    "general_distill_6layer_30h_from9layerTA-experiment_epoch1000000_step179332"
#  

#   "bert_base"
#   
#   "general_distill_9layer_36h-experiment_epoch1000000_step141206"
#   

}

os.environ["WANDB_DISABLED"] = "true"

def train(device, config):

    cmd = f'CUDA_VISIBLE_DEVICES={device} python run_glue.py ' 
    options = []
    for k, v in config.items():   
        if v is not None:
            options += [ f'{k} {v}']  
        else: options += [ f'{k}']



    cmd += ' '.join(options)

    os.system(cmd)


def experiment_start(task, device, batch_size, lr, path, group):
    config = finetune_config
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%R")
    config['--log_group'] = group
    config['--model_name_or_path'] = path
    

    if task in ['cola', 'sst2', 'rte', 'mrpc']:
        epoch = 3
        config['--eval_steps'] = 50
    else:
        epoch =10
    save_dir = f"runs/{group}/{task}/{lr}lr_{batch_size}bz_{epoch}epoch"
    config['--num_train_epochs'] = epoch
    config['--learning_rate'] = lr
    config['--output_dir'] = save_dir
    config['--task_name'] = task
    config['--per_device_train_batch_size'] = batch_size

    config['--run_name'] = f'{task}_{lr}lr_{batch_size}bz_{epoch}epoch'

    train(device, config)




model_list = [
    # ("~/code/academic-budget-bert/training-out/general_distill_6layer_30h-experiment/experiment/epoch1000000_step150619","general_distill_6layer_30h-experiment_epoch1000000_step150619" ),
    # ("~/code/academic-budget-bert/training-out/general_distill_6layer_60h_continue-experiment/experiment/epoch1000000_step301541","general_distill_6layer_60h_continue-experiment_epoch1000000_step301541")
    # ( "~/code/academic-budget-bert/training-out/general_distill_9layer_36h-experiment/experiment/epoch1000000_step141206","general_distill_9layer_36h-experiment_epoch1000000_step141206")
    # ("~/code/academic-budget-bert/training-out/general_distill_6layer_30h_lastlayer_teacher-experiment/experiment/epoch1000000_step174459", "general_distill_6layer_30h_lastlayer_teacher-experiment"),
    # ("~/code/academic-budget-bert/training-out/pretraining_experiment-/epoch1000000_step210494", "bertbase")
    ("training-out/general_distill_6layer_30h_twostage_attval_mlm-experiment-experiment/experiment/epoch1000000_step232546", "general_distill_6layer_30h_twostage_attval_mlm")
]



num_workers = 4


args = []

for model in model_list:

    # for task in ['cola', 'qnli', 'sst2', 'mnli', 'qqp', 'rte', 'cola', 'mrpc']: 
    for task in ['rte']:
    # for task in ["rte"]:
        
        idx = 0
        for lr in [1e-5, 3e-5, 5e-5, 8e-5]:
            for bz in [8]:#, 16, 32]:
                args.append((task, idx, bz, lr, model[0], model[1]))
                idx += 1

writer_workers = Pool(num_workers)
writer_workers.starmap(experiment_start, args)



# experiment_start('cola', 0, 16,1e-5)