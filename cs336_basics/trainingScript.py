import argparse
import torch
import numpy as np
import wandb
import time

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.train_model import cross_entropy, AdamW, cosine_lr_schedule_with_warmup,\
                                    gradient_clipping, data_loading, save_checkpoint, load_checkpoint


def get_args():

    #创建解析器对象
    parser=argparse.ArgumentParser(description='Train a Transformer LM')
    #模型超参数
    parser.add_argument('--vocab_size',type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--theta', type=float, default=10000.0, help='float for RoPE')
    #优化器超参数
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='betas for AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for AdamW')
    parser.add_argument('--lr_max', type=float, default=1e-3, help='max learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-8, help='min learning rate')
    parser.add_argument('--eps', type=float, default=1e-6, help='epsilon for AdamW')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    #训练相关
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warm_iters', type=int, default=200)
    parser.add_argument('--total_iters', type=int, default=10000)
    parser.add_argument('--log_interval', type=int, default=10) # 每隔多少步打印一次训练日志（如 loss、准确率）
    parser.add_argument('--eval_interval', type=int, default=200) # 每隔多少步在验证集上做一次评估
    parser.add_argument('--eval_iters', type=int, default=50) # 做验证时，用多少个 batch 的验证集数据来估算 loss/准确率
    parser.add_argument('--checkpoint_interval', type=int, default=1000) # 每隔多少步保存一次模型权重
    #路径
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume_from', type=str, default=None)
    #训练记录
    parser.add_argument('--wandb_project', type=str, default='cs336_lm')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    return parser.parse_args()



def main():
    args = get_args()

    # 用 memmap 而不是直接 np.load,避免整个数据集读进内存
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args) # 把超参数记录下来，方便之后对比不同run
        )

    # 模型，优化器初始化
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
        betas=args.betas,
        eps=args.eps
    )

    # inference所需参数，即模型设置参数
    hp_for_infer={
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'num_layers': args.num_layers,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_heads': args.num_heads,
        'theta': args.theta
    }

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    start_time = time.time()  # 记录训练开始的时间戳

    for it in range(start_iter, args.total_iters):
        # 学习率调度(cosine schedule with warmup)
        lr = cosine_lr_schedule_with_warmup(it, args.lr_max, args.lr_min,
                                     args.warm_iters, args.total_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr

        inputs, targets = data_loading(train_data, args.batch_size,
                                        args.context_length, args.device)

        logits = model(inputs)  # (batch_size, seq_len, vocab_size)
        # 交叉熵本质上是"对每一个token位置独立算一次loss，然后求平均
        loss = cross_entropy(
            logits.reshape(-1, logits.shape[-1]),   # -> (batch_size*seq_len, vocab_size)
            targets.reshape(-1)                     # -> (batch_size*seq_len,)
        )

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if it % args.log_interval == 0:
            elapsed = time.time() - start_time  # wall-clock time(秒)
            print(f"iter {it}: train loss {loss.item():.4f}, lr {lr:.6f}, elapsed {elapsed:.1f}s")
            if args.use_wandb: 
                wandb.log({'train_loss': loss.item(), 
                           'lr': lr,
                           'train/wall_clock_time': elapsed,
                }, step=it)  # step=it 这样x轴默认就是gradient step

        if it % args.eval_interval == 0:
            val_loss = estimate_val_loss(model, val_data, args)
            elapsed = time.time() - start_time
            print(f"iter {it}: val loss {val_loss:.4f}")
            if args.use_wandb:
                wandb.log({
                    'val_loss': val_loss,
                    'val/wall_clock_time': elapsed,
                }, step=it)

        if it % args.checkpoint_interval == 0 or it == args.total_iters - 1:
            save_checkpoint(model, optimizer, hp_for_infer, it, args.checkpoint_path)

    if args.use_wandb:
        wandb.finish()


@torch.no_grad()
def estimate_val_loss(model, val_data, args):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        inputs, targets = data_loading(val_data, args.batch_size,
                                        args.context_length, args.device)
        logits = model(inputs)
        losses.append(cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


if __name__ == '__main__':
    main()
