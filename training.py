import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

# ========================
# Set up Argument Parser
# ========================
parser = argparse.ArgumentParser(description='Demonstration of training a model with a Transformer')
parser.add_argument('-bs', type=int, required=True, help='Provide the batch_size as an integer')
args = parser.parse_args()
print(f'Batch size: {args.bs}')

# ========================
# Set up Device and Hyperparameters
# ========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device used: {device}')

batch_size    = args.bs       # Batch size during training
block_size    = 64            # Context length (number of tokens per block)
learning_rate = 3e-4          # Learning rate for the optimizer
max_iters     = 4000          # Maximum training iterations
eval_iters    = 100           # Evaluation frequency (number of iterations between evaluations)
n_embd        = 384           # Token embedding dimension
n_layer       = 8             # Number of Transformer blocks
n_head        = 8             # Number of heads in multi-head self-attention
dropout       = 0.2           # Dropout rate

# ========================
# Read and Create Vocabulary from File
# ========================
with open('dataset/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # Get a sorted list of unique characters
    chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'Vocabulary Size: {vocab_size}')

# Create mappings from string to integer and vice versa
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
# Encoding function: convert a string to a list of integers
encode = lambda s: [string_to_int[c] for c in s]
# Decoding function: convert a list of integers to a string
decode = lambda i: ''.join([int_to_string[index] for index in i])

# ========================
# Utility Function to Read Dataset Using Memory Mapping
# ========================
def get_random_chunk(split):
    """
    Returns a random chunk of text from the dataset file.
    Uses memory mapping for efficiency when handling large files.
    
    Args:
        split (str): 'train' or 'val' to choose the appropriate split file.
        
    Returns:
        Tensor of integer data (encoded text).
    """
    filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open('dataset/' + filename, 'rb') as f:
        # Memory-map the file to avoid loading the whole file into RAM
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            # Choose a random start position so that the text block varies
            start_pos = random.randint(0, file_size - block_size * batch_size)
            mm.seek(start_pos)
            # Read a text block of length (batch_size * block_size - 1)
            block = mm.read(block_size * batch_size - 1)
            # Decode bytes to string, ignoring decode errors and removing carriage returns
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            # Encode the string into integer tensor
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data

def get_batch(split):
    """
    Generates a mini-batch for training or validation.
    
    Args:
        split (str): 'train' or 'val'
        
    Returns:
        Tuple of (x, y) where x is the input and y is the target (offset by one token)
    """
    data = get_random_chunk(split)
    # Randomly select starting indices for sequences of length block_size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Collect input sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Targets are the input sequences shifted by one token
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    """
    Estimates the average loss for both 'train' and 'val' splits.
    Evaluates over several iterations for a more stable estimate.
    
    Returns:
        Dictionary with keys 'train' and 'val' and their respective average losses.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ========================
# Debugging: Example Data Chunk Extraction
# (This part is only for debugging and viewing data samples)
# ========================
data_sample = get_random_chunk('train')
x_sample = data_sample[:block_size]
y_sample = data_sample[1:block_size+1]
for t in range(0, block_size):
    context = x_sample[:t+1]
    target  = y_sample[t]
    print(f'Input: {context} -> Target: {target}')

# ========================
# Transformer Model Definition
# ========================

class Head(nn.Module):
    """
    A single head of self-attention (scaled dot-product attention).
    """
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Create a lower-triangular mask to prevent "peeking" into the future
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Project input to key, query, and value vectors
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Compute attention scores (scaled dot product)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # Apply the lower-triangular mask so that each token only attends to previous tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Normalize the scores with softmax to get probabilities
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Project to value vector and aggregate
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention: combining several attention heads in parallel.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Linear layer to combine outputs from all heads
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs from each head
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """
    A simple feed-forward layer after self-attention and residual connection.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand dimension
            nn.ReLU(),                      # Non-linearity
            nn.Linear(4 * n_embd, n_embd),   # Project back to the original dimension
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    A Transformer block comprising MultiHeadAttention, FeedForward,
    and corresponding residual connections with LayerNorm.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Self-attention with residual connection and LayerNorm
        y = self.sa(x)
        x = self.ln1(x + y)
        # Feed-forward with residual connection and LayerNorm
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class JPLanguageModel(nn.Module):
    """
    A language generation model based on a Transformer.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table     = nn.Embedding(vocab_size, n_embd)  # Token embeddings
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) # Positional embeddings
        # Stack Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)  # Final LayerNorm
        # Linear layer to map to vocabulary size (before softmax)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _init_weight(self, module):
        # Optional weight initialization for model layers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        """
        Forward pass of the model.
        
        Args:
            index (Tensor): (B, T) sequence of tokens
            targets (Tensor, optional): target sequence for computing loss
        
        Returns:
            logits: Model output predictions
            loss: Cross-entropy loss if targets are provided; otherwise, None
        """
        B, T = index.shape
        # Get token embeddings and add positional embeddings
        tok_emb = self.token_embedding_table(index)  # (B, T, n_embd)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb
        # Pass through stacked Transformer blocks
        x = self.blocks(x)
        # Final normalization
        x = self.ln_f(x)
        # Map to vocabulary output
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for cross-entropy loss computation
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        """
        Text generation function.
        
        Args:
            index (Tensor): (B, T) tensor of starting token indices as context.
            max_new_tokens (int): Number of new tokens to generate.
        
        Returns:
            Tensor: The extended sequence with newly generated tokens.
        """
        for _ in range(max_new_tokens):
            # Limit the context to at most block_size tokens
            index_cond = index if index.size(1) <= block_size else index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            # Focus on the last time step's logits
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Convert logits to probabilities
            prob = F.softmax(logits, dim=-1)
            # Sample the next token from the probability distribution
            index_next = torch.multinomial(prob, num_samples=1)  # (B, 1)
            # Concatenate the new token to the sequence
            index = torch.cat((index, index_next), dim=1)
        return index

# ========================
# Initialize Model, Optimizer, and Setup Mixed Precision (AMP)
# ========================
model = JPLanguageModel(vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize GradScaler for mixed precision training
scaler = torch.amp.GradScaler(device=device)

# Set gradient accumulation steps (e.g., accumulate over 4 mini-batches to save memory)
gradient_accumulation_steps = 4

# ========================
# Training Loop
# ========================
model.train()
for iter in range(max_iters):
    # Evaluate loss at defined intervals to monitor training progress
    if (iter % eval_iters == 0):
        losses = estimate_loss()
        print(f"Iteration: {iter}, Training Loss: {losses['train']:.5f}, Validation Loss: {losses['val']:.5f}")
    
    optimizer.zero_grad(set_to_none=True)
    
    # Accumulate gradients over multiple mini-batches
    for acc_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        # Use autocast for mixed precision for improved efficiency
        with torch.amp.autocast(device_type=device):
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps  # Normalize loss according to accumulation steps

        scaler.scale(loss).backward()
    
    # Update model parameters with the scaled gradients
    scaler.step(optimizer)
    scaler.update()

print("Training finished. Final loss:", loss.item())

# ========================
# Save the Model to a File Using Pickle
# ========================
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved successfully.')
