import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# ============================================
# 1. TOKENIZER (Simple character-level)
# ============================================
class SimpleTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

# ============================================
# 2. DATASET
# ============================================
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.data = tokenizer.encode(text)
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

# ============================================
# 3. TRANSFORMER COMPONENTS
# ============================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape
        
        # Linear projections
        Q = self.query(x)  # (batch, seq_len, embed_size)
        K = self.key(x)
        V = self.value(x)
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        # Causal mask (prevent looking ahead)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)
        out = self.fc_out(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, embed_size)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        attention_out = self.attention(x)
        x = self.norm1(x + self.dropout(attention_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

# ============================================
# 4. MAIN LLM MODEL
# ============================================
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, 
                 ff_hidden_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_seq_len, embed_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
        
    def forward(self, idx):
        batch_size, seq_len = idx.shape
        
        # Token embeddings
        tok_emb = self.token_embedding(idx)  # (batch, seq_len, embed_size)
        
        # Position embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)  # (seq_len, embed_size)
        
        # Combine embeddings
        x = tok_emb + pos_emb
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate text given a starting sequence"""
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx[:, -self.max_seq_len:]
            
            # Get predictions
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Focus on last token
            
            # Apply softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# ============================================
# 5. TRAINING FUNCTION
# ============================================
def train_model(model, train_loader, epochs, lr, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# ============================================
# 6. MAIN EXECUTION
# ============================================
def main():
    # Sample text (in practice, use much more data!)
    text = """
    To be or not to be, that is the question.
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them.
    """ * 100  # Repeat for more training data
    
    # Hyperparameters
    BLOCK_SIZE = 64      # Context length
    BATCH_SIZE = 32
    EMBED_SIZE = 128     # Embedding dimension
    NUM_HEADS = 4        # Attention heads
    NUM_LAYERS = 4       # Transformer layers
    FF_HIDDEN_SIZE = 512 # Feed-forward hidden size
    DROPOUT = 0.1
    LEARNING_RATE = 3e-4
    EPOCHS = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    tokenizer = SimpleTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    dataset = TextDataset(text, tokenizer, BLOCK_SIZE)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        embed_size=EMBED_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_hidden_size=FF_HIDDEN_SIZE,
        max_seq_len=BLOCK_SIZE,
        dropout=DROPOUT
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining...")
    train_model(model, train_loader, EPOCHS, LEARNING_RATE, device)
    
    # Generate text
    print("\nGenerating text...")
    model.eval()
    prompt = "To be"
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=100, temperature=0.8)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {tokenizer.decode(generated[0].tolist())}")

if __name__ == "__main__":
    main()
