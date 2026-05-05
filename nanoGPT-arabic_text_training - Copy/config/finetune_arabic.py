import time

out_dir = 'out-arabic'
eval_interval = 5
eval_iters = 40
wandb_log = False

dataset = 'arabic'
init_from = 'gpt2'

always_save_checkpoint = False

batch_size = 4
gradient_accumulation_steps = 8
max_iters = 100
block_size = 512

learning_rate = 3e-5
decay_lr = False

device = 'cuda'
dtype = 'bfloat16'
compile = False
