## Currently: Chat bot work testing with Discord Channel ##
## Originally started with Andrej Karpathy's tutorial: https://www.youtube.com/watch?v=kCc8FmEb1nY with additional notes where I have them. ##
## For learning purposes toward building vector Language Models for time-series forecasting, agents, ##
## and other applications. ##
## I know I suck. Don't @ me. :P ##

## Next Steps: ##
## 1. Understanding the encoder architecture and how it works. Add to model.## 
## 2. Research and build backend memory for the model. ##
## 3. Panel/UI package for model training and evaluation. ##

## 5/2/2023 - Added load_checkpoint function to model.py,

import torch
import torch.nn as nn
from datetime import datetime, date
from torch.nn import functional as F
import pandas as pd
import csv
from csv import DictWriter

print(f'Cuda is available? : {torch.cuda.is_available()}!')

#hyperparameters
#### 5/4/23 - CREATE INIT FILE HERE FOR PARAMETER PASSING ####
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
load_model = True
now = datetime.now()
date_time = now.strftime("%m_%d_%y_%H:%M:%S")
print(f'Model Start: {now.strftime("%m/%d/%Y, %H:%M:%S")}')

corpus_only = True
evaluation_mode = False

#-----------------#
### Seed selection is used in LLM models. Update this to save seeds based on context. ###
torch.manual_seed(1337)

## GATHER TEXT DATA ##

if evaluation_mode:
    text = input('Enter text to evaluate: ')
    text = text.lower()

else:
    if corpus_only:
        data_path = 'GPT/inputs/corpus_only.txt'
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        data_path = 'GPT/inputs/input_preprocessed.txt'
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

## Create Checkpoint Function##
def checkpoint(state, filename=f'GPT/checkpoints/checkpoint.pth.tar'):
    print(f'Saving checkpoint to {filename}...')
    torch.save(state, filename)
    print(f'Checkpoint saved.')

## Load Last Checkpoint Function##
def load_checkpoint(checkpoint, model):
    print(f'Loading checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    #model.optimizer.load_state_dict(checkpoint['optimizer']) - OPTIMIZER FOR CHECKPOINT?!
    print(f'Checkpoint loaded.')
    return model

def get_filename_datetime():
    return f'export_{str(date.today())}.txt'



## Get all unique characters in text ##
chars = sorted(list(set(text)))
vocab_size = len(chars)
## Create a mapping from characters to integers ##
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


## Encode Text ##
## Use for both training and evaluation ##


print('Encoding Text')
data = torch.tensor(encode(text), dtype=torch.long, device=device)

# Train/test split for training. Eval is only 
if evaluation_mode:
    pass
else:
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

# Data loading
def get_batch(split):
    # generate small batch of data inputs x/targets y
    #### ENCODING EVAL BREAKS HERE 5/4/2023 ####
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y - x.to(device), y.to(device)
    return x, y

@torch.no_grad() #Tells Pytorch to not call backward/backprop on the loss
def estimate_loss():
    print('Estimating loss...')
    # 4/22/2023 Adding Checkpoint block to save model state

    # if load_model:
    #     load_checkpoint(torch.load(f'GPT/checkpoints/checkpoint.pth.tar'), model)
    # 4/22/2023 End Checkpoint block
    out = {}
    print('Evalutating Model...')
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    print('Training Model')
    model.train()
    return out

class Head(nn.Module):
    """  One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @(B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # (B, T, T)
        # perform weighted aggregation of values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


#Multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):
    ''' Multi-head self-attention '''

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # (B, T, C*n_heads)
        #out = self.proj(out) # (B, T, C)
        return out

# Feed Forward module
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Block of self-attention + feed forward
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size= n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) #Layer Norm before self-attention
        self.ln2 = nn.LayerNorm(n_embd) #Layer Norm before feed forward
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # (Batch,Time,Channel) One head of self-attention
        x = x+ self.ffwd(self.ln2(x)) # (Batch,Time,Channel) Feed Forward module
        return x

# Model class used for training
class ChatModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        #self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) - Optimizer load for checkpoint?? 5/2/2023
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dimensional self-attention
        #self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        #idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (Batch,Time,Channel(Vocab Size))
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (Time,Channel)
        x = tok_emb + pos_emb # (Batch,Time,Channel)
        x = self.blocks(x) # (Batch,Time,Channel) One head of self-attention
        #x = self.ffwd(x) # (Batch,Time,Channel) Feed Forward module
        logits = self.lm_head(x) # (Batch,Time,Vocab Size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # or -1
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get predictions

            idx_cond = idx[:, -block_size:] # (B, T) -> (B, block_size)
            logits, loss = self(idx_cond) #loss is ignored currently
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = ChatModel()
m = model.to(device)
if evaluation_mode:
    checkpoint = {'state_dict': model.state_dict()}
    load_checkpoint(torch.load(f'GPT/checkpoints/checkpoint.pth.tar'), model)
    m.eval()
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    export = decode(m.generate(context, max_new_tokens=200)[0].tolist())
    print(f'CodyAI: {export}')
else:
    if load_model:
        checkpoint = {'state_dict': model.state_dict()}
        load_checkpoint(torch.load(f'GPT/checkpoints/checkpoint.pth.tar'), model)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if evaluation_mode:
    pass
    # context = torch.zeros((1,1), dtype=torch.long, device=device)
    # export = decode(m.generate(context, max_new_tokens=200)[0].tolist())
    # print(f'CodyAI: {export}')

######## MODEL EVALUATION LOOP ########
else:
    for iter in range(max_iters):
        # occasionally evaluate the loss on train and val sets
        print(f'iter: {iter}, datetime: {now.strftime("%m-%d-%y_%H:%M:%S")}')
        if iter % eval_interval == 0:
    #### MOVED SAVE CHECKPOINT TO HERE 5/3/2023 ####
            print('Saving checkpoint...')
            torch.save(checkpoint, f'GPT/checkpoints/checkpoint.pth.tar')
            print('Checkpoint saved...')
    #### MOVED SAVE CHECKPOINT TO HERE 5/3/2023 ####
            losses = estimate_loss()
            now = datetime.now()
            print(f'time: {now.strftime("%m/%d/%y %H:%M:%S")};iter {iter}; train loss: {losses["train"]:.3f}; val loss: {losses["val"]:.3f}')
            # save validation loss metrics
            with open('GPT/validation/losses.csv', 'a') as csv_file:
                field_names = ['datetime','epoch','train_losses','val_losses','learning_rate','batch_size','block_size','embed_size','num_heads','num_layers']
                now = datetime.now()
                dict = {'datetime': now.strftime("%m-%d-%y_%H:%M:%S"),'epoch':iter,'train_losses':losses["train"],'val_losses':str(losses["val"]), 'learning_rate':learning_rate, 'batch_size':batch_size, 'block_size':block_size, 'embed_size':n_embd, 'num_heads':n_head, 'num_layers':n_layer}
                dictwriter_object = DictWriter(csv_file, fieldnames=field_names) 
                dictwriter_object.writerow(dict)
                csv_file.close()

        # sample a batch of data
        xb, yb = get_batch('train')
        # forward pass
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# generate from the model

if evaluation_mode:
    pass
else:
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    name = get_filename_datetime()
    with open (f'GPT/validation/{name}', 'w', encoding='utf-8') as text_file:
        print('Generating text...')
        export = decode(m.generate(context, max_new_tokens=200)[0].tolist())
        print('Exporting text...')
        text_file.writelines(export)
        print('Text exported to {text_file.name}')

# save the model

if evaluation_mode:
    pass
else:
    print('Saving model...')
    if corpus_only:
        torch.save(model.state_dict(), f'GPT/validation/model-corpus-only.pt')
    else:
        torch.save(model.state_dict(), f'GPT/validation/model.pt')