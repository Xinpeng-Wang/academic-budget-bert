import os
import sys
import collections
import torch
from multiprocessing import Pool
import yaml
from pathlib import Path
from datetime import datetime
from ruamel.yaml import YAML
yaml = YAML(typ='rt')
yaml.preserve_quotes = True
config_path = 'config/pretrain.yaml'
config = yaml.load(Path(config_path))

cmd = 'CUDA_VISIBLE_DEVICES=MIG-feee0c61-26ec-5616-bb59-b6d777cc4f34,MIG-c0386dcc-84bf-55a0-bb83-93d415e7a9e3  python -m torch.distributed.launch run_pretraining.py '

options = []
for k, v in config.items():   
    if v != 'None':
        options += [ f'{k} {v}']  
    else: options += [ f'{k}']


cmd += ' '.join(options)
job_name = config['--job_name']
run_id = config['--current_run_id']
log_path = f'training-out/{job_name}-{run_id}'
if not os.path.exists(log_path):
    os.mkdir(log_path)
yaml.dump(config, Path(config_path))
os.system(cmd)
