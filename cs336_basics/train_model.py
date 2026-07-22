import torch
def cross_entropy(logits:torch.tensor, targets:torch.tensor):
    """
    logits: 模型输出的原始分数, (batch_size, vocab_size)
    targets: i.e. 序列位置i对应的正确答案的token_id, (batch_size,)
    """
    logits=logits-torch.max(logits,dim=-1,keepdim=True).values # 保证数值稳定，防止exp过大
    exp_logits=torch.exp(logits)
    targets_lgts=exp_logits[torch.arange(logits.shape[0]), targets] # (batch_size,)

    sum=exp_logits.sum(dim=-1)  # (batch_size,)
    softmax=targets_lgts/sum    # (batch_size,)
    ce= -(softmax.log())

    loss=ce.mean(dim=0)         
    return loss
