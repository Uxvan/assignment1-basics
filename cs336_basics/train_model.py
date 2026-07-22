import torch
def cross_entropy(logits:torch.tensor, targets:torch.tensor):
    """
    logits: 模型输出的原始分数, (batch_size, seq_len, vocab_size)
    targets: i.e. 每个序列位置i对应的正确答案的token_id, (batch_size, seq_len)
    """
    logits=logits-torch.max(logits,dim=-1,keepdim=True).values() # 保证数值稳定，防止exp过大
    exp_logits=torch.exp(logits)
    targets_lgts=exp_logits[targets] # (batch_size, seq_len)

    sum=exp_logits.sum(dim=-1)  # (batch_size, seq_len)
    softmax=targets_lgts/sum    # (batch_size, seq_len)
    ce= -(softmax.log())

    loss=ce.mean(dim=0)         # (seq_len,)
    return loss

def perplexity(loss):
    p=torch.exp(loss.mean())
    return p