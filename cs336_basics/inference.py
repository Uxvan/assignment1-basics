import torch
import argparse

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM, softmax

tokenizer = Tokenizer.from_files(
    vocab_filepath='train_results/vocab_merges/vocab_tinystories.pkl',
    merges_filepath='train_results/vocab_merges/merges_tinystories.pkl',
    special_tokens=['<|endoftext|>'],
)

def get_args():
    parser = argparse.ArgumentParser()
    #加载checkpoint权重
    parser.add_argument('--checkpoint_path', type=str, required=True)
    #推理参数
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model_for_inference(args):
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    config_for_infer=checkpoint['config_for_infer']

    model = TransformerLM(
        vocab_size=config_for_infer['vocab_size'],
        context_length=config_for_infer['context_length'],
        d_model=config_for_infer['d_model'],
        num_layers=config_for_infer['num_layers'],
        num_heads=config_for_infer['num_heads'],
        d_ff=config_for_infer['d_ff'],
        theta=config_for_infer['theta'],
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()  # 关键:关闭 dropout 等训练专用行为
    return model, config_for_infer


def top_p_sampling(x:torch.Tensor, p: float=0.9) -> torch.Tensor:
    """x: (batch_size, vocab_size) temperature_softmax的结果"""
    sorted_prob, sorted_idx=torch.sort(x, descending=True, dim=-1)
    cumsum_prob=torch.cumsum(sorted_prob, dim=-1)

    mask=(cumsum_prob-sorted_prob) > p
    sorted_prob[mask]=0
    sorted_sum=sorted_prob.sum(dim=-1, keepdim=True)
    normed_prob=sorted_prob/sorted_sum


    sorted_token_idx=torch.multinomial(normed_prob, 1) # 按降序排列的概率，从中按概率随机选一个token概率值的索引
    next_token_id = torch.gather(sorted_idx, dim=-1, index=sorted_token_idx) # 上面索引在sorted_idx对应选中token的token_id
    return next_token_id


@torch.no_grad() # 推理不需要梯度,省内存、加速
def inference(model, input_ids, max_new_tokens: int, context_length: int, temperature: float=1.0, top_p: float=0.9, 
              eos_token_id=None, device='cpu'):
    
    input_ids=input_ids.to(device) # (batch_size, input_seq_len),通常推理时batch_size=1
    model.eval()

    for _ in range(max_new_tokens):
        idx=input_ids[:, -context_length:] # 做截断防止input过长超过模型承载, context_length相当于max_seq_len

        logits=model(idx) #生成概率, (batch=1, input_seq_len, vocab_size)
        logits=logits[:,-1,:] # 只要最后一个位置 → (batch=1, vocab_size)

        res=softmax(logits/temperature, -1) #temperature_softmax
        next_token_id=top_p_sampling(res, top_p) # top_p 采样

        input_ids=torch.cat([input_ids, next_token_id], dim=-1) #把新生成的token_id加进序列，方便下一轮inference

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
                    break
    return input_ids

def main():
    args=get_args()
    model, config_for_infer=load_model_for_inference(args)

    # encode 返回list[int]，要转换为tensor
    prompt_ids=tokenizer.encode(args.prompt)
    input_ids=torch.tensor([prompt_ids], dtype=torch.long) #(batch, input_seq_len），这里batch=1

    eos_token_id=tokenizer.encode('<|endoftext|>')[0]
    output_ids=inference(
        model, input_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=config_for_infer['context_length'],
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        device=args.device,
    )

    # output_ids是torch.Tensor,shape是(batch=1, seq_len),但tokenizer.decode通常期望输入是list[int]
    generated_text = tokenizer.decode(output_ids[0].tolist()) 
    print(generated_text) 


if __name__=='__main__':
    main()
