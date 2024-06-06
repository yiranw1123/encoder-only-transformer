import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.d_model = config['d_model']
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model) # vocab_size * d_model
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].requires_grad_(False)
    

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config["d_model"], config['d_ff'])
        self.linear2 = nn.Linear(config["d_ff"], config['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        d_model = config['d_model']
        num_heads = config['num_attention_heads']
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = config['d_model']
        self.num_heads = config['num_attention_heads']
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.layer_norm_1 = nn.LayerNorm(self.d_model)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(x, x, x)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.num_layers = config['num_hidden_layers']
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(self.num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerForSentenceClassification(nn.Module):

    def __init__(self, config, vocab_size):
        super().__init__()
        d_model = config['d_model']
        num_labels = config['num_labels']
        max_seq_length = config['max_seq_len']

        self.tokens_embedding = InputEmbedding(config, vocab_size)
        self.positional_embedding = PositionalEncoding(d_model, max_seq_length)
        self.encoder = TransformerEncoder(config, vocab_size=vocab_size)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.classifier = nn.Linear(d_model, num_labels)


    def forward(self, x):
        x = self.tokens_embedding(x)
        x = self.positional_embedding(x)
        x = self.encoder(x)[:, 0 ,:]
        x = self.dropout(x)
        x = self.classifier(x)
        return x