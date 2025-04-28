import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import time
import argparse

parser = argparse.ArgumentParser(description='This is a demonstration program')
parser.add_argument('-bs', type=str, required=True, help='Please provide a batch_size')
args = parser.parse_args()
print(f'batch size: {args.bs}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = args.bs
block_size = 64
learning_rate = 3e-4
max_iters = 4000
eval_iters = 100
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2
print(device)

chars = ""
with open('dataset/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)

string_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_string = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda i: ''.join([int_to_string[index] for index in i])
    
class Head(nn.Module):
    ''' One Head Self Attention or Head Class or Scaled Dot Product Attention '''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**(-0.5) # (B,T,hs) @ (B,hs,T) --> (B,T,T)
        # Use dynamic tril mask instead of fixed self.tril
        # tril = torch.tril(torch.ones(T, T, device=dvc))
        # wei = wei.masked_fill(tril == 0, float('-inf')) # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B,T,T) @ (B,T,hs) --> (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    '''Self Attention with Multi-Head so that can parallel'''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,C) or (B,T,F) dim=-1 equal to last dimension. (B,T,F) ---> (B,T,[h1,h1,h1,h1,h2,h2,h2,h2,h3,h3,h3,h3])
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    ''' FeedForward step After Multihead Attention and Add&Norm '''
    ''' A Simple Linear Layer followed by a non-linearity '''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    ''' Transformer Blocks '''
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: total head on Multi-head Attention with Mask
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class JPLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # input embedding
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) # positional encoding/embedding, bisa menggunakan rumus fixed atau nn.Embedding
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # Decoder blocks that running sequentially atau biasa disebut Transformer Block

        self.ln_f = nn.LayerNorm(n_embd) # final layer norm (disebelah kanan decoder akhir, yaitu sebelum nn.Linear)
        self.lm_head = nn.Linear(n_embd, vocab_size) # tahap nn.Linear disebelah kanan decoder akhir

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        # logits = self.token_embedding_table(index) ### Error ###
        B, T = index.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # Batch size, Time steps, Channel
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context.
        for _ in range(max_new_tokens):
            index_cond = index if index.size(1) <= block_size else index[:, -block_size:]
            # get the preddiction
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] #becomes (B, C), get for All Kalimat(semua matriks), Semua Token Terkakhir(baris terakhir), dan Semua Dimension Embedding(semua kolom)
            prob = F.softmax(logits, dim=-1)# (B, C), last dimension of shape, example (2,3), it mean Columns
            # sample from the distribution
            index_next = torch.multinomial(prob, num_samples=1) # (B, 1), get 1 sample for each row
            # append sample index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1), gabungkan/tambahkan ke dalam kolom
        return index

model = JPLanguageModel(vocab_size)
m = model.to(device)

print('loading model.....')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('load successfully.....')
m = model.to(device)

while True:
    prompt = input("Input: ")
    if prompt == "exit":
        break
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')