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
        epoch = 10
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

# ("training-out/general_6layer_30h_from_base_last_att_val_no_fuse_1024bz-experiment/experiment/epoch2767_step408953","post/general_6layer_30h_from_base_last_att_val_no_fuse_1024bz/epoch2767_step408953")
# ("training-out/general_6layer_30h_from_base_skip_att_val-experiment/experiment/epoch3399_step124322","post/general_6layer_30h_from_base_skip_att_val/epoch3399_step124322")
# ("training-out/general_9layer_30h_from_base_last_att_val_nofuse_5e-4-experiment/experiment/epoch774_step114636","post/general_9layer_30h_from_base_last_att_val_nofuse_5e/epoch774_step114636")
# ("training-out/general_6layer_30h_from_base_skip_att_val-experiment/experiment/epoch3820_step139697", "post/general_6layer_30h_from_base_skip_att_val/epoch3820_step139697/apex")
# ("training-out/pre-ln/general_distill_6layer_60h_continue-experiment/experiment/epoch1000000_step301541","pre/general_distill_6layer_60h_continue/epoch1000000_step301541")
# ("training-out/pre-ln/general_distill_6layer_30h-experiment/experiment/epoch1000000_step150619", "pre/general_distill_6layer_30h/epoch1000000_step150619")
# ("training-out/general_6layer_30h_from_base_last_att_val_no_fuse_1024bz-experiment/experiment/epoch3688_step545046","post/general_6layer_30h_from_base_last_att_val_no_fuse_1024bz/epoch3688_step545046")
# ("training-out/general_9layer_30h_from_base_last_att_val_nofuse_5e-4-experiment/experiment/epoch1163_step172068",'post/general_9layer_30h_from_base_last_att_val_nofuse_5e-4/epoch1163_step172068')
# ("training-out/6layer_from_base_last_1024_512length_30h_5e-4-experiment/experiment/epoch647_step36396", "post/6layer_from_base_last_1024_512length_30h_5e-4/epoch647_step36396")
# ("training-out/6layer_from_base_last_1024_512length_30h_5e-4-experiment/experiment/epoch1297_step72796","post/6layer_from_base_last_1024_512length_30h_5e-4/epoch1297_step72796")
# ("training-out/fail_reproduce_2-experiment/experiment/epoch436_step64866","post/fail_reproduce_2/epoch436_step64866")
# ("training-out/6layer_from_base_last_1024_512length_30h_5e-4-experiment/experiment/epoch3351_step188102","post/6layer_from_base_last_1024_512length_30h_5e-4/epoch3351_step188102")
# ("training-out/test-mse/mse/epoch767_step43208","post/test-mse/epoch767_step43208")
# ("/dss/dsshome1/lxc01/di75wud/code/academic-budget-bert/training-out/general_6layer_30h_from_base_skip_att_val-experiment/experiment/epoch425_step15616","post/general_6layer_30h_from_base_skip_att_val/epoch425_step15616")
# ("training-out/att_val_3072_512_1e-3/3/epoch1879_step35075","post/att_val_3072_512_1e-3/epoch1879_step35075")
# ("training-out/att_val_256_512_5e-4_fp16apex_dsinf_preln_kl32/1/epoch719_step161988", 'post/att_val_256_512_5e-4_fp16apex_dsinf_preln_kl32/epoch719_step161988')
# ("training-out/6layer_from_base_last_1024_512length_30h_5e-4-experiment/experiment/epoch1946_step109223", 'post/6layer_from_base_last_1024_512length_30h_5e-4-experiment/epoch1946_step109223')
# ('training-out/att_val_3072_512_1e-3/3/epoch4334_step80872', 'post/att_val_3072_512_1e-3/epoch4334_step80872')
# ('training-out/att_val_256_512_5e-4_fp16apex_dsinf_preln_kl32/1/epoch1569_step352972', 'post/att_val_256_512_5e-4_fp16apex_dsinf_preln_kl32/epoch1569_step352972')
("training-out/pre-ln/general_6layer_10h_from_base_lastlayer_pearcol-experiment/experiment/epoch1000000_step167145", "pre/general_6layer_10h_from_base_lastlayer_pearcol-experiment/epoch1000000_step167145"),
("training-out/InitMinilm_6layer_30h_from_base_last_att_val_nofuse_5e-4-experiment/experiment/epoch2146_step317102", "post/InitMinilm_6layer_30h_from_base_last_att_val_nofuse_5e-4-experiment/epoch2146_step317102"),
("training-out/pre-ln/general_distill_6layer_30h_twostage_attval_mlm-experiment-experiment/experiment/epoch1000000_step232546", "pre/general_distill_6layer_30h_twostage_attval_mlm-experiment/epoch1000000_step232546")
]



num_workers = 0


args = []

for model in model_list:
    idx = 0
    for task in ['cola','rte','mnli', 'qqp','qnli','sst2', 'mrpc']:
    # for task in ['rte']:
    # for task in ["rte"]:
        if idx == 8:
            idx = 0
        for lr in [1e-5, 3e-5, 5e-5, 8e-5]:
            bz=32
            if task == 'rte':
                bz=16
            args.append((task, idx, bz, lr, model[0], model[1]))
            num_workers +=1 
            idx+=1

    writer_workers = Pool(num_workers)
    writer_workers.starmap(experiment_start, args)

# for model in model_list:

#     for task in ['cola']:#[ 'rte','mnli', 'qqp','qnli','sst2', 'cola', 'mrpc']:
#     # for task in ['rte']:
#     # for task in ["rte"]:
#         lr=3e-5
        
#         idx = 1
#         bz = 32
#         if task == 'rte':
#             bz=16
#         args.append((task, idx, bz, lr, model[0], model[1]))
#         num_workers +=1 
#         # idx+=1



# experiment_start('cola', 0, 16,1e-5)


# pre-ln
# ("~/code/academic-budget-bert/training-out/general_distill_6layer_60h_continue-experiment/experiment/epoch1000000_step301541","general_distill_6layer_60h_continue-experiment_epoch1000000_step301541")
# ("~/code/academic-budget-bert/training-out/general_distill_6layer_30h-experiment/experiment/epoch1000000_step150619","general_distill_6layer_30h-experiment_epoch1000000_step150619" ),
# ( "~/code/academic-budget-bert/training-out/general_distill_9layer_36h-experiment/experiment/epoch1000000_step141206","general_distill_9layer_36h-experiment_epoch1000000_step141206")
# ("~/code/academic-budget-bert/training-out/general_distill_6layer_30h_lastlayer_teacher-experiment/experiment/epoch1000000_step174459", "general_distill_6layer_30h_lastlayer_teacher-experiment"),
# ("~/code/academic-budget-bert/training-out/pretraining_experiment-/epoch1000000_step210494", "bertbase")
# ("training-out/general_distill_6layer_30h_twostage_attval_mlm-experiment-experiment/experiment/epoch1000000_step232546", "general_distill_6layer_30h_twostage_attval_mlm")
# ("training-out/general_12layer_30h_from_bertlarge_lastlayer-experiment/experiment/epoch1000000_step78084",  "general_12layer_30h_from_bertlarge_lastlayer")
# ("training-out/general_6layer_10h_from_base_lastlayer_pearcol-experiment/experiment/epoch2287_step83627", "general_6layer_10h_from_base_lastlayer_pearcol_epoch2287_step83627")
# ("training-out/bert_base/bert_base", "bert_base")
# ("training-out/general_6layer_10h_from_base_lastlayer_pearcol-experiment/experiment/epoch1000000_step167145","general_6layer_10h_from_base_lastlayer_pearcol_epoch1000000_step167145")
