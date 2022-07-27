import os
from datetime import datetime

# train config 
finetune_config = {
    "--model_name_or_path": "bert-base-cased",
  "--task_name": "mrpc",
  "--do_train": None,
  "--do_eval": None,
  "--max_seq_length": 128,
  "--per_device_train_batch_size": 32,
  "--learning_rate": 2e-5,
  "--num_train_epochs": 5,
  "--output_dir": "runs/mrpc/"
}


# define command function
def train(device, config):
    cmd = f'CUDA_VISIBLE_DEVICES={device} python $HOME/transformers/examples/pytorch/text-classification/run_glue.py ' 
    options = []
    for k, v in config.items():   
        if v is not None:
            options += [ f'{k} {v}']  
        else: options += [ f'{k}']
    cmd += ' '.join(options)
    os.system(cmd)


def start_experiment(device, task, lr):
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%R")
    config = finetune_config
    config["--learning_rate"] = lr
    config["--task_name"] = task
    config["--output_dir"] = f'/mounts/work/xinpeng/runs_models/academic_budget/runs/{task}/{current_time}'
    config['--model_name_or_path'] = '~/MiniLM-L6-H768-distilled-from-BERT-Base'
    train(device, config)

# loop over task    
for task in ['qqp']:
    for lr in [2e-5]:
        start_experiment(0, task, lr)
    # run command