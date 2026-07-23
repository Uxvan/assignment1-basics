import torch
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
    def __init__(self, params, alpha, beta_1, beta_2, gamma, eps):
        defaults=dict(alpha=alpha, beta_1=beta_1, beta_2=beta_2, eps=eps, gamma=gamma)
        super().__init__(params,defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()

        for group in self.param_groups:
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
                group['alpha']=group['alpha'] * (1-group['beta_2']**t).sqrt() / (1-group['beta_1']**t)
                p -= group['alpha']*group['gamma']*p
                state['m']=state['m']*group['beta_1']+(1-group['beta_1'])*grad
                state['v']=state['v']*group['beta_2']+(1-group['beta_2'])*grad*grad
                p -= group['alpha']*state['m']/((state['v']).sqrt()+group['eps'])

        return loss
