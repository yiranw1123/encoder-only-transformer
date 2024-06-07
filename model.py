import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps: float = 10 ** -6)-> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocab_size, d_model) # vocab_size * d_model
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].requires_grad_(False)
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()

        assert d_model % h == 0, "d_model must be divisible by num_heads"
        
        self.h = h
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(Q,K,V, mask, dropout:nn.Dropout):
        d_k = Q.shape[-1]

        attention_scores = (Q @ K.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e-9)
        attention_scores = attention_scores.softmax(dim = -1) #(batch, h, seq_len, d_k)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ V), attention_scores


    def forward(self, Q, K, V, mask=None):
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)

        #(batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        Q = Q.view(Q.shape[0], Q.shape[1], self.h, self.d_k).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.h, self.d_k).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(Q,K,V,mask, self.dropout)
        #(batch, seq_len, h, d_k) ---> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) ->  None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class ClassificationLayer(nn.Module):
    def __init__(self, d_model, num_labels):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x):
        return self.classifier(x)

class SentenceClassificationTransformer(nn.Module):
    def __init__(self, encoder, src_embd, src_pos_embd, classifier):
        super().__init__()
        self.encoder = encoder
        self.src_embd = src_embd
        self.src_pos_embd = src_pos_embd
        self.classifier = classifier
    
    def forward(self, src, src_mask):
        src = self.src_embd(src)
        src = self.src_pos_embd(src)
        src = self.encoder(src, src_mask)[:, 0, :]
        src = self.classifier(src)
        return src