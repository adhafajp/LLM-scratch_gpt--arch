import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
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
print(f'Vocab size: {vocab_size}')

string_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_string = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda i: ''.join([int_to_string[index] for index in i])

#memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open('dataset/' + filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            #determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            #seek to the random position and read the block text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size - 1)

            #decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            #train test split
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
        # print(file_size)
        # print(start_pos)
        # print(block)
        # print(decoded_block)
    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix]) # example [[1, 2, 3], [2, 3, 4, [3, 4, 5]]
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # [[2, 3, 4], [3, 4, 5], [4, 5, 6]]
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y) # ======= logits, loss = model.__call__(X, Y) ======= logits, loss = model.forward(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

data_data = get_random_chunk('train')
x = data_data[:block_size]
y = data_data[1:block_size+1]

for t in range(0, block_size):
    context = x[:t+1]
    target = y[t]
    print(f'when input is {context} target is {target}')
    
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

# # load model
# print('loading model.....')
# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('load successfully.....')
# m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    if (iter % eval_iters == 0) :
        losses = estimate_loss()
        print(f"Iter: {iter}, Train Loss: {losses['train']:.5f}, Validation Loss: {losses['val']:.5f}")
    optimizer.zero_grad(set_to_none=True)
    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)
    loss.backward() #backprop hitung gradien dari loss
    optimizer.step() #update parameter dari gradien yang sudah dihitung backprop
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved success.....')