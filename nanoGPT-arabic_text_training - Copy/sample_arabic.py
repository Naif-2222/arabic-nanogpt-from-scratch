import os
import pickle
import torch
from transformers import AutoTokenizer
from model import GPTConfig, GPT




'''
start = "بسم الله"                    # "In the name of God"
# or
start = "المملكة العربية السعودية"   # "The Kingdom of Saudi Arabia"  
# or
start = "في القرن العشرين"            # "In the twentieth century"

start = "لبنك السعودي للاستثمار
'''
out_dir        = 'out-arabic-aragpt2'
start          = 'في عام'
num_samples    = 3
max_new_tokens = 200
temperature    = 0.8
top_k          = 40
device         = 'cuda'
dtype          = 'bfloat16'

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
model = GPT(gptconf)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)

with open(os.path.join('data', 'arabic', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
tokenizer = AutoTokenizer.from_pretrained(meta['tokenizer_id'])

ids = tokenizer.encode(start)
x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad(), ctx:
    for _ in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(tokenizer.decode(y[0].tolist()))
        print('---')