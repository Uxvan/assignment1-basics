import torch
from collections.abc import Iterable

def cross_entropy(logits:torch.tensor, targets:torch.tensor):
    """
    logits: 模型输出的原始分数, (batch_size, vocab_size)
    targets: i.e. 序列位置i对应的正确答案的token_id, (batch_size,)
    """
    logits=logits-torch.max(logits,dim=-1,keepdim=True).values # 保证数值稳定，防止exp过大
    exp_logits=torch.exp(logits)

    targets_lgts=logits[torch.arange(logits.shape[0]), targets] # (batch_size,)
    sum=exp_logits.sum(dim=-1)  # (batch_size,)

    loss=sum.log() - targets_lgts    # (batch_size,),不直接log是为了防止某个logit对应值过小，导致下溢出现-inf
    loss=loss.mean(dim=0)         
    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params :Iterable[torch.nn.Parameter],
                lr: float, betas :tuple[float, float],
                weight_decay: float, eps: float):
        defaults=dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps) # 
        super().__init__(params,defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()

        for group in self.param_groups:
            lr=group['lr'] # alpha
            beta1, beta2=group['betas']
            weight_decay=group['weight_decay'] # gamma
            eps=group['eps']

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad=p.grad.data
                state=self.state[p]

                if len(state)==0:
                    state["steps"]=0
                    state['m']=torch.zeros_like(p)
                    state['v']=torch.zeros_like(p)

                state["steps"]+=1
                t=state["steps"]

                lr_t = lr * (1 - beta2**t).sqrt() / (1 - beta1**t)
                p -= lr * weight_decay * p
                state['m'] = state['m'] * beta1 + (1 - beta1) * grad
                state['v'] = state['v'] * beta2 + (1 - beta2) * grad * grad
                p -= lr_t * state['m'] / ((state['v']).sqrt() + eps)

        return loss
