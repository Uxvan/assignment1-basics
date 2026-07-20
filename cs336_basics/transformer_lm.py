import torch
import torch.nn as nn
from einops import einsum
from collections import OrderedDict

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_dim=in_features
        self.out_dim=out_features
        self.weight=nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        self.device=device
        self.dtype=dtype
        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result= x @ self.weight
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
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: # token_positions:(..., seq_len)
        choosen_sin=self.sin_value[token_positions]                                    # x:(..., seq_len,d_k)
        choosen_cos=self.cos_value[token_positions]
        x_odd=x[...,0::2]
        x_even=x[...,1::2]
        rotated_odd=x_odd*choosen_cos-x_even*choosen_sin
        rotated_even=x_odd*choosen_sin+x_even*choosen_cos # (max_seq_len, d_k//2)
        rotated_x=torch.stack([rotated_odd,rotated_even],dim=-1).flatten(-2)
        return rotated_x
    

def softmax(x: torch.Tensor, i: int): # i-th dimension
    m=torch.max(x,dim=i, keepdim=True)
    x_stable=x-m #减去最大值使数值稳定, 防止e^x过大
    exp_x=torch.exp(x_stable)
    sum=exp_x.sum(dim=i)
    result=exp_x/sum
    return result


def scaled_dot_product_attention(Q, K, V, mask=None): 
    '''
    Q,K:(batch_size, ..., seq_len, d_k) ; 
    V:(batch_size, ..., seq_len, d_v); 
    mask:(seq_len, seq_len)

    Attention(𝑄, 𝐾, 𝑉 ) = softmax(𝑄𝐾^𝑇/√𝑑𝑘 )𝑉
    '''
    d_k=Q.size[-1]                                    
    dot_product=einsum(
        Q,K,
        '... seq_len d_k, ... seq_len d_k  ->  ... seq_len seq_len'
    )/d_k.sqrt()

    if mask:
        dot_product[mask==False]=float('-inf')

    attention=einsum(
        softmax(dot_product,-1), V,
        '... seq_len seq_len, ... seq_len d_v -> ... seq_len d_v'
    )
    return attention


class MultiHeadselfAttention(nn.Module):  
    '''
    MultiHeadSelfAttention(𝑥) = 𝑊_𝑂 MultiHead(𝑊_𝑄𝑥, 𝑊_𝐾𝑥, 𝑊_𝑉𝑥) 
    𝑊_𝑄 (ℎ𝑑𝑘, 𝑑model), 𝑊_𝐾 (ℎ𝑑𝑘, 𝑑model), 𝑊_𝑉 (ℎ𝑑𝑣, 𝑑model) , 𝑊_𝑂 (𝑑model, ℎ𝑑𝑣)
    d_model: Dimensionality of the Transformer block inputs;
    num_heads: Number of heads
    '''                                    
    def __init__(self, d_model:int, num_heads:int, theta:float, max_seq_len:int): 
        super().__init__() 
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k, self.d_v=d_model/num_heads       # 等同于head_dim
        self.W_Q=nn.Linear(self.d_k,d_model,bias=False)
        self.W_K=nn.Linear(self.d_k,d_model,bias=False)
        self.W_V=nn.Linear(self.d_v,d_model,bias=False)
        self.W_O=nn.Linear(d_model,self.d_v,bias=False)
        self.rope=RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

    def casual_masking(self,max_seq_len):
        x=torch.ones((max_seq_len,max_seq_len))
        mask=torch.triu(x, diagonal=-1).bool()

    def forward(self, x, token_positions, mask=None):    # x:(...,seq_len,d_model)
        seq_len=x.shape[-2]
        mask=self.casual_masking(seq_len)

        x.reshape(*x.shape[:-1],self.num_heads,-1)      # x:(...,seq_len,num_heads,head_dim)
        Q, K, V=self.W_Q(x), self.W_K(x), self.W_V(x)

        RoPE_Q=self.rope(Q,token_positions)
        RoPE_K=self.rope(K,token_positions)

        multi_atten=scaled_dot_product_attention(RoPE_Q,RoPE_K,V,mask)
        multi_atten=multi_atten.reshape(*x.shape[:-2], self.d_model)
        out_atten=self.W_O(multi_atten)

        return out_atten


class TransformerBlock(nn.Module):

    def __init__(self, d_model:int, num_heads:int, d_ff:int, theta:float, max_seq_len:int, token_positions:torch.tensor):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.token_positions=token_positions
        self.norm=RMSNorm(self.d_model)
        self.mha=MultiHeadselfAttention(d_model, num_heads, theta, max_seq_len)
        self.ff=PositionwiseFeedforward(d_ff)

    def forward(self, x):
        y=self.norm(x)
        out1= x+ self.mha(y,self.token_positions)
       
        z=self.norm(out1)
        out2= out1 +self.ff(z, self.ff(z))   # out2: (..., seq_len, d_model)

        return out2
    

class TransformerLM(nn.Module):
    '''
    vocab_size: The size of the vocabulary, necessary for determining the dimensionality of the
token embedding matrix.
    context_length: The maximum context length, necessary for determining the dimensionality
of the RoPE sin and cos buffer.
    num_layers: The number of Transformer blocks to use.
    '''

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, 
                 d_model:int, num_heads:int, d_ff:int, theta:float, 
                 token_ids:torch.tensor, token_positions:torch.tensor):
        super().__init__()
        self.num_embeddings=vocab_size
        self.max_seq_len=context_length
        self.num_layers=num_layers

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.theta=theta
       
        self.token_ids=token_ids
        self.token_positions=token_positions
        
        self.token_embedding=Embedding(vocab_size, d_model)
        self.block=TransformerBlock(d_model, num_heads, d_ff, theta, context_length,token_positions)
        self.norm=RMSNorm(d_model)
        self.linear=Linear(d_model,d_model)

    def forward(self,x):

        layers=[]
        for _ in range(self.num_layers):
            layers.append(self.block)
        
        transformBlocks=nn.Sequential(*layers)

        x1=self.token_embedding(x,self.token_ids)
        x2=transformBlocks(x1)
        x3=self.norm(x2)
        x4=self.linear(x3)
        out=softmax(x4, -1)

        return out





