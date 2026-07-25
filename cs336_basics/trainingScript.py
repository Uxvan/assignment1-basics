import argparse
import torch
import numpy as np

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.train_model import cross_entropy, AdamW, cosine_lr_schedule_with_warmup,\
                                    gradient_clipping, data_loading, save_checkpoint, load_checkpoint

def get_args():
    #创建解析器对象
    parser=argparse.ArgumentParser(description='Train a Transformer LM')
    #模型超参数
    parser.add_argument('--vocab_size',type=int, required=True)
    parser.add_argument('--context_length', type=int, required=256)
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

    return parser.parse_args()



def main():
    args = get_args()

    # 用 memmap 而不是直接 np.load,避免整个数据集读进内存
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

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

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    for it in range(start_iter, args.total_iters):
        # 学习率调度(cosine schedule with warmup)
        lr = cosine_lr_schedule_with_warmup(it, args.lr_max, args.lr_min,
                                     args.warmup_iters, args.total_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr

        inputs, targets = data_loading(train_data, args.batch_size,
                                        args.context_length, args.device)

        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.log_interval == 0:
            print(f"iter {it}: train loss {loss.item():.4f}, lr {lr:.6f}")
            # 如果用 wandb: wandb.log({'train_loss': loss.item(), 'lr': lr}, step=it)

        if it % args.eval_interval == 0:
            val_loss = estimate_val_loss(model, val_data, args)
            print(f"iter {it}: val loss {val_loss:.4f}")

        if it % args.checkpoint_interval == 0 or it == args.total_iters - 1:
            save_checkpoint(model, optimizer, it, args.checkpoint_path)


@torch.no_grad()
def estimate_val_loss(model, val_data, args):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        inputs, targets = data_loading(val_data, args.batch_size,
                                        args.context_length, args.device)
        logits = model(inputs)
        losses.append(cross_entropy(logits, targets).item())
    model.train()
    return sum(losses) / len(losses)


if __name__ == '__main__':
    main()