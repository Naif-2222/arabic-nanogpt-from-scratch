import time

out_dir = 'out-arabic-aragpt2'
eval_interval = 500
eval_iters = 100
log_interval = 25
wandb_log = False

dataset = 'arabic'
init_from = 'scratch'
always_save_checkpoint = False

# GPT-2 small architecture, ~110M params
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2
bias = False

block_size = 512
batch_size = 12
gradient_accumulation_steps = 11     # effective batch = 128 sequences

max_iters = 2000
learning_rate = 3e-4
warmup_iters = 200
lr_decay_iters = 20000
min_lr = 3e-5
decay_lr = True
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

device = 'cuda'
dtype = 'bfloat16'
compile = False