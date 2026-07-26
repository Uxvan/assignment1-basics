import torch
import torch.nn as nn
from einops import einsum
from collections import OrderedDict


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_dim=in_features
        self.out_dim=out_features
        self.weight=nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.device=device
        self.dtype=dtype
        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result= einsum(
            self.weight, x,
            'out_dim in_dim, ... in_dim -> ... out_dim'
        )
        return result
    def init_weight(self):
        std=(2/(self.in_dim+self.out_dim))**0.5
        nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)
        


class Embedding(nn.Module):
    '''
    num_embeddings: int Size of the vocabulary, i.e. vocab_size
    embedding_dim: int Dimension of the embedding vectors, i.e., d_model

    forward方法中, 传入形状为(batch_size, sequence_length)的torch.LongTensor(里面存的是token_id),
    然后用它去索引一个(vocab_size, d_model)的嵌入矩阵, 这样为每个token ID取出对应嵌入向量
    '''
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.d_model=embedding_dim
        self.embed_matrix=nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.device=device
        self.dtype=dtype
        self.init_matrix()

    def init_matrix(self):
        nn.init.trunc_normal_(self.embed_matrix,mean=0,std=1,a=-3,b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed_vectors=self.embed_matrix[token_ids]      # (batch_size, seq_len, d_model)
        return embed_vectors 
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.g=nn.Parameter(torch.ones(self.d_model))
        self.device=device
        self.dtype=dtype
    
    def forward(self, x: torch.Tensor): #x:(batch_size, sequence_length, d_model)
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=1/(((x*x).sum(dim=-1)/self.d_model+self.eps).sqrt())
        normed_x=einsum(
            x,rms,
            'batch_size sequence_length d_model, batch_size sequence_length -> batch_size sequence_length d_model' 
        )
        result=einsum(
            normed_x,self.g,
            'batch_size sequence_length d_model, d_model -> batch_size sequence_length d_model'
        )
        result=result.to(in_dtype)
        return result
    

class PositionwiseFeedforward(nn.Module):
    '''
     d_ff: Dimensionality of the position-wise feed-forward inner layer.
    '''
    def __init__(self, d_model, d_ff:int):     
        super().__init__()
        self.d_ff=d_ff
        self.d_model=d_model
        self.w1=nn.Linear(d_model,d_ff,bias=False) #nn.Linear内部权重和bias自动保存在nn.Parameters中
        self.w2=nn.Linear(d_ff,d_model,bias=False)
        self.w3=nn.Linear(d_model,d_ff,bias=False)
    def forward(self,x): # FFN(𝑥) = SwiGLU(𝑥, 𝑊1, 𝑊2, 𝑊3) = 𝑊2 (SiLU(𝑊1𝑥) ⊙ 𝑊3𝑥),
        y1=self.w1(x)
        silu= y1 * torch.sigmoid(y1)
        y3=self.w3(x)
        y2=silu * y3
        ffn=self.w2(y2)
        return ffn
    

class RotaryPositionalEmbedding(nn.Module):
    '''
    d_k: dimension of query and key vectors
    seq_len表示token个数
    max_seq_len表示模型配置的"最大可能序列长度"
    '''
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
        super().__init__()
        theta_freq=1/theta**( torch.arange(0,d_k,2)/d_k ) # (d_k//2,)
        angle=einsum(
            torch.arange(0,max_seq_len),theta_freq,
            'i, k -> i k'
        ) # (max_seq_len, d_k//2)
        sin_value=torch.sin(angle).to(device)
        cos_value=torch.cos(angle).to(device)
        self.register_buffer('sin_value',sin_value)
        self.register_buffer('cos_value',cos_value)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor =None) -> torch.Tensor: # token_positions:(..., seq_len), x:(..., seq_len,d_k)
        if token_positions==None:
            token_positions=torch.arange(x.shape[-2])
        choosen_sin=self.sin_value[token_positions].unsqueeze(-3)               # (..., seq_len, d_k//2) -> (..., 1, seq_len, d_k//2), 方便多头注意力时num_heads维的广播
        choosen_cos=self.cos_value[token_positions].unsqueeze(-3) 
        x_odd=x[...,0::2]
        x_even=x[...,1::2]
        rotated_odd=x_odd*choosen_cos-x_even*choosen_sin
        rotated_even=x_odd*choosen_sin+x_even*choosen_cos # (max_seq_len, d_k//2)
        rotated_x=torch.stack([rotated_odd,rotated_even],dim=-1).flatten(-2)
        return rotated_x
    

def softmax(x: torch.Tensor, i: int): # i-th dimension
    m=torch.max(x,dim=i, keepdim=True).values
    x_stable=x-m #减去最大值使数值稳定, 防止e^x过大
    exp_x=torch.exp(x_stable)
    sum=exp_x.sum(dim=i,keepdim=True)
    result=exp_x/sum
    return result


def scaled_dot_product_attention(Q, K, V, mask=None): 
    '''
    Q,K:(batch_size, ..., seq_len, d_k) ; 
    V:(batch_size, ..., seq_len, d_v); 
    mask:(seq_len, seq_len)

    Attention(Q, K, V) = softmax(QK^T/√𝑑𝑘 )V
    '''
    d_k=Q.shape[-1]                                    
    dot_product=einsum(
        Q,K,
        '... seq_len_q d_k, ... seq_len_k d_k  ->  ... seq_len_q seq_len_k'     # 给两个 seq_len 分别起名, 否则einsum会报错
    )/(d_k**0.5)

    if mask is not None:       # 不用 'if mask', 防止张量真值歧义
        dot_product= dot_product.masked_fill(mask==False, float('-inf')) # 不用 dot_product[mask==False]=float('-inf')原地修改，
                                                                         # 防止autograd计算图出错

    attention=einsum(
        softmax(dot_product,-1), V,
        '... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v'
    )
    return attention


class MultiHeadselfAttention(nn.Module):  
    '''
    MultiHeadSelfAttention(x) = W_O MultiHead(W_Qx, W_Kx, W_Vx) 
    W_Q (h*dk, dmodel), W_K (h*dk, dmodel), W_V (h*dv, dmodel) , W_O (dmodel, h*dv)
    d_model: Dimensionality of the Transformer block inputs;
    num_heads: Number of heads
    '''                                    
    def __init__(self, d_model:int, num_heads:int, theta:float=None, max_seq_len:int=None, device=None): 
        super().__init__() 
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k, self.d_v=d_model//num_heads, d_model//num_heads       # 等同于head_dim
        self.W_Q=nn.Linear(d_model,d_model,bias=False)
        self.W_K=nn.Linear(d_model,d_model,bias=False)
        self.W_V=nn.Linear(d_model,d_model,bias=False)
        self.W_O=nn.Linear(d_model,d_model,bias=False)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device) if theta is not None else None


    def forward(self, x, token_positions=None):    # x:(...,seq_len,d_model)
        seq_len=x.shape[-2]
        # 创建mask
        casual_mask=torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

        Q, K, V=self.W_Q(x), self.W_K(x), self.W_V(x)
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.d_v).transpose(-2,-3)      #  -> (...,seq_len,num_heads,head_dim) -> (...,num_heads,seq_len,head_dim)

        if self.rope is not None:
            Q=self.rope(Q,token_positions)
            K=self.rope(K,token_positions)

        multi_atten=scaled_dot_product_attention(Q, K, V, casual_mask)     # (...,num_heads,seq_len,d_v)
        multi_atten=multi_atten.transpose(-2,-3).reshape(*x.shape[:-1], self.d_model)
        out_atten=self.W_O(multi_atten)

        return out_atten


class TransformerBlock(nn.Module):

    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len:int, theta:float, token_positions=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.max_seq_len=max_seq_len
        self.theta=theta
        self.token_positions=token_positions
        self.norm1=RMSNorm(d_model)
        self.norm2=RMSNorm(d_model)
        self.mha=MultiHeadselfAttention(d_model, num_heads, theta, max_seq_len)
        self.ffn=PositionwiseFeedforward(d_model,d_ff)

    def forward(self, x):
        y=self.norm1(x)
        out1= x+ self.mha(y, self.token_positions)
       
        z=self.norm2(out1)
        out2= out1 +self.ffn(z)   # out2: (..., seq_len, d_model)

        return out2
    

class TransformerLM(nn.Module):
    '''
    vocab_size: The size of the vocabulary, necessary for determining the dimensionality of the
token embedding matrix.
    context_length: The maximum context length, necessary for determining the dimensionality
of the RoPE sin and cos buffer. i.e. max_seq_len
    num_layers: The number of Transformer blocks to use.
    '''

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, 
                 d_model:int, num_heads:int, d_ff:int, theta:float):
        super().__init__()
        self.num_embeddings=vocab_size
        self.max_seq_len=context_length
        self.num_layers=num_layers

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.theta=theta
               
        self.token_embedding=Embedding(vocab_size, d_model)
        self.norm=RMSNorm(d_model)
        self.linear=Linear(d_model, vocab_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_layers)
        ])

    def forward(self,x):

        x1=self.token_embedding(x)
        x2=x1
        for layer in self.layers:
            x2=layer(x2)
        x3=self.norm(x2)
        x4=self.linear(x3)

        return x4





