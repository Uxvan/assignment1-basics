import torch
from collections.abc import Iterable
import math
import numpy as np


def cross_entropy(logits:torch.tensor, targets:torch.tensor):
    """
    logits: 模型输出的原始分数, (batch_size, vocab_size)
    targets: i.e. 序列位置i对应的正确答案的token_id, (batch_size,)
    """
    logits=logits-torch.max(logits,dim=-1,keepdim=True).values # 保证数值稳定，防止exp过大
    exp_logits=torch.exp(logits)

    targets_lgts=logits[torch.arange(logits.shape[0]), targets] # (batch_size,)
    sum=exp_logits.sum(dim=-1)  # (batch_size,)

    loss=sum.log() - targets_lgts  # (batch_size,),不直接log是为了防止某个logit对应值过小，导致下溢出现-inf
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

                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p -= lr * weight_decay * p
                state['m'] = state['m'] * beta1 + (1 - beta1) * grad
                state['v'] = state['v'] * beta2 + (1 - beta2) * grad * grad
                p -= lr_t * state['m'] / ((state['v']).sqrt() + eps)

        return loss


def cosine_lr_schedule_with_warmup(t, lr_max, lr_min, tw, tc):
    if t < tw:
        lr_t= t/tw * lr_max
    elif tw<= t <=tc:
        lr_t= lr_min + 1/2 * (1 + math.cos((t-tw)/(tc-tw) * torch.pi)) * (lr_max - lr_min)
    else:
        lr_t=lr_min
    return lr_t


def gradient_clipping(params: Iterable[torch.nn.Parameter], M: float, eps: float=1e-6):
    total_norm=0
    params_withgrad=[]
    for p in params:
        if p.grad is None:
            continue
        params_withgrad.append(p)

    total_norm= math.sqrt(sum(p.grad.pow(2).sum() for p in params_withgrad))
    if total_norm > M:
        clip_coef= M/(total_norm + eps)
        for p in params_withgrad:
            p.grad.mul_(clip_coef)


def data_loading(x: np.ndarray, batch_size, context_length, 
                 device: str| torch.device)-> tuple[torch.Tensor, torch.Tensor]:
    # 随机采样batch_size个长度为context_length的窗口
    max_start=len(x) - context_length
    start_idx=np.random.randint(0, max_start, batch_size)
    inputs=np.stack([x[i:i+context_length] for i in start_idx])
    targets=np.stack([x[i+1:i+1+context_length] for i in start_idx])
    inputs=torch.from_numpy(inputs).long().to(device)
    targets=torch.from_numpy(inputs).long().to(device) #.long()把 Tensor 转换成 torch.int64
    return inputs,targets
