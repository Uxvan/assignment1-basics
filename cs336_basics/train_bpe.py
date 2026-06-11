import regex as re
import re as builtin_re

def train_bpe(input_path, vocab_size, special_tokens):
    # 1. 初始化基础词表 (0-255 字节)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    
    # 2. 分配特殊 Token 的 ID (紧跟在 255 后面)
    start_ID = 256
    for tok in special_tokens:
        vocab[start_ID] = tok.encode('utf-8')
        start_ID += 1

    # GPT-2/GPT-4 核心正则表达式
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正确处理特殊标记：如果存在特殊标记，用正则安全切分
    if special_tokens:
        special_pattern = '|'.join(builtin_re.escape(tok) for tok in special_tokens)
        # 用捕获括号 () 确保 split 结果保留特殊标记
        parts = builtin_re.split(f'({special_pattern})', content)
    else:
        parts = [content]

    # 构建基础序列数据
    byte_data = []
    for p in parts:
        if not p:
            continue
        if p in special_tokens:
            # 特殊 Token 作为独立的一个整体，不参与文本切分与内部合并
            # 这里我们在训练语料中可以不处理它，或者把它当作一个整体的原子ID（这里不放入byte_data参与BPE合并）
            continue
        
        # 正常文本切分
        pieces = re.findall(PAT, p)
        for piece in pieces:
            byte_data.append(list(piece.encode('utf-8')))

    # 3. 初次统计全局的字节对频次
    pair_count = {}
    for piece in byte_data:
        for i in range(len(piece) - 1):
            pair = (piece[i], piece[i + 1])
            pair_count[pair] = pair_count.get(pair, 0) + 1

    # 4. 核心 BPE 循环
    while len(vocab) < vocab_size:
        if not pair_count:
            break
            
        # 寻找最高频的字节对
        max_count = max(pair_count.values())
        max_pairs = [k for k, v in pair_count.items() if v == max_count]
        
        # 如果有同频的，选择字节序最大的（按你的原逻辑）
        merge_pair = max(max_pairs) 

        # 写入词表
        vocab[start_ID] = vocab[merge_pair[0]] + vocab[merge_pair[1]]
        merges.append((vocab[merge_pair[0]], vocab[merge_pair[1]]))

        # 动态更新数据和频次（采用重构重建法，避免指针移位 Bug）
        new_pair_count = {}
        new_byte_data = []
        
        for piece in byte_data:
            if len(piece) < 2:
                new_byte_data.append(piece)
                continue
                
            new_piece = []
            i = 0
            while i < len(piece):
                # 如果匹配到要合并的对
                if i < len(piece) - 1 and (piece[i], piece[i+1]) == merge_pair:
                    new_piece.append(start_ID)
                    i += 2 # 跳过被合并的右元素
                else:
                    new_piece.append(piece[i])
                    i += 1
            
            new_byte_data.append(new_piece)
            
            # 重新统计这个 piece 产生的新 pair
            for j in range(len(new_piece) - 1):
                p = (new_piece[j], new_piece[j+1])
                new_pair_count[p] = new_pair_count.get(p, 0) + 1
                
        byte_data = new_byte_data
        pair_count = new_pair_count
        
        start_ID += 1
    
    return vocab, merges
'''
import regex as re
import re as builtin_re

def train_bpe(input_path,vocab_size,special_tokens):

    vocab = {i: bytes([i]) for i in range(256)}
    merges=[]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_pattern='|'.join(builtin_re.escape(tok) for tok in special_tokens) 

    with open(input_path,'r') as f:
        content=f.read()

    parts=builtin_re.split(f'{special_pattern}',content) #把文本按照special_tokens分割为大块,同时special_tokens作为独立元素保留在结果
    normal_parts=[p for p in parts if p not in special_tokens] 

    pieces=[]
    for p in normal_parts:
        piece=re.findall(PAT,p) #在大分割里小分割，按照空格和标点，由每一大块得到['hello','world',...]
        pieces.extend(piece) #pieces中元素为word
    byte_data = [list(piece.encode('utf-8')) for piece in pieces] #[[piece1],[piece2],...]，此时每一个最内的列表(piece)表示一个word的token, eg.[[104,105],[...],...]，[104,105]对应'h','i'
    
    start_ID=256
    for i in special_tokens:
        vocab[start_ID]=i.encode('utf-8')
        start_ID+=1

    #初始化字节对
    pair_count={}
    for piece in byte_data:
        for i in range(len(piece)-1):
            pair=(piece[i],piece[i+1])
            pair_count[pair]=pair_count.get(pair,0)+1 # {(pair):count,...}

    while len(vocab)<vocab_size:
        
        max_count=max(pair_count.values()) #最大频率
        max_pairs=[k for k,v in pair_count.items() if v==max_count] #一列有最大出现频率的pairs
        merge_pair=max(max_pairs) #选择最大字节序

        vocab[start_ID]=vocab[merge_pair[0]]+vocab[merge_pair[1]] #eg.['h','i'] -> ['hi'],注意要先把token转换为字符
        merges.append((vocab[merge_pair[0]],vocab[merge_pair[1]]))

        #更新pair_count
        pair_count.pop(merge_pair)
        for piece in byte_data:
            i=0
            while i<len(piece)-1:
                if (piece[i],piece[i+1])==merge_pair:
                    if i+2<len(piece):#减去右邻pair,加上新右邻
                        pair_count[(piece[i+1],piece[i+2])]-=1 #以merge_pair=ab为例，减去[d,a,b,c]中的pair(b,c)个数
                        pair_count[(start_ID,piece[i+2])]=pair_count.get((start_ID,piece[i+2]),0)+1 #添加(ab,c)
                    if  i>0:#减去旧左邻,加上新左邻
                        pair_count[(piece[i-1],piece[i])]-=1
                        pair_count[(piece[i-1],start_ID)]=pair_count.get((piece[i-1],start_ID),0)+1
                        
                    #更新byte_data
                    piece[i]=start_ID
                    del piece[i+1]
                        
                i+=1
        start_ID+=1
    
    return vocab,merges


    

import os
import collections
from typing import List, Tuple, Dict, Set
import json
import regex    
from collections import defaultdict
import pickle

def merge_token_sequence(token_seq: Tuple, best_pair: Tuple, new_token: bytes) -> Tuple:
    """在一个token序列中，将所有出现的 best_pair 合并为 new_token"""
    new_seq = []
    i = 0
    while i < len(token_seq):
        # 检查当前位置是否是最佳对的开始
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == best_pair:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(token_seq[i])
            i += 1
    return tuple(new_seq)

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # 第0步要先校验一下参数，为了更好地增强函数的鲁棒性
    if not isinstance(vocab_size,int) or vocab_size <= 0:
        raise ValueError("vocab_size 必须是一个正整数。")

    # 第1步初始化词汇表，基础词汇表包含所有256个基础字节，对应ASCII码范围是0-255
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}#bytes()是将整数转换为字节序列的函数，bytes(3)=bytes([0,0,0])即只传数字就是构造三个0，如果传数字列表bytes([65,66])=b'AB'返回ASCII码，注意范围是0-255
    current_next_id: int = 256 # 新的token ID从256开始

    # token_frequency_table: Dict[Tuple[bytes], int] = {} # 用于统计每个token出现的频率，注意不能用列表，智能用tuple元组，因为列表不可哈希
    token_frequency_table = defaultdict(int) #总是存在没出现过的key，只好用defaultdict
    # 用一个集合来高效检查特殊符号的字节表示是否已存在于词汇表中，用列表也能查重，但时间复杂度是O(n)，集合是O(1)
    existing_byte_values: Set[bytes] = set(vocab.values())

    # 添加特殊符号到词汇表
    for st_str in special_tokens:
        if len(vocab) >= vocab_size: # 如果词汇表满了，就不再添加
            break
        st_bytes = st_str.encode("utf-8") # 将特殊符号字符串转为字节串
        if st_bytes not in existing_byte_values: # 只有当这个字节串不在现有词汇中时才添加（避免重复，例如特殊符号 "a" 和基础字节 b'a'）
            vocab[current_next_id] = st_bytes # 将新的字节串添加到词汇表中
            existing_byte_values.add(st_bytes) # 记录这个新的字节值
            current_next_id += 1 # 更新下一个token ID


    # 第2步加载训练的语料库       
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read() # 读取整个文件内容
    except FileNotFoundError:
        text = "" # 如果文件不存在，视为空文本处理


    # 第3步对语料库里的文段进行预分词pre-tokenization：分割文本时保存标点和空格，得到“单词”列表['Hello', ',', ' world', '!', ' This', ' is', ' a', ' test', '.']
    chunks = regex.split('|'.join(map(regex.escape,special_tokens)),text) #首先按照特殊字符进行大分割，比如<endoftext>按照章节分割
    # 然后在大分割里小分割，按照空格和标点
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        for word in regex.findall(PAT, chunk):
            word_bytes = word.encode("utf-8") #对每一个单词进行编码，并转换为bytes
            bytes_list = [bytes([x]) for x in word_bytes] #e.g. ['h', 'e', 'l', 'l', 'o']
            token_frequency_table[tuple(bytes_list)] += 1 #统计每个token出现的频率


    merges: List[Tuple[bytes, bytes]] = [] # 用于存储合并操作记录

    #一次性统计所有token的对和频率
    pair_counts = defaultdict(int)
    for token in token_frequency_table.keys():
        for i in range(len(token) - 1):
            pair_counts[token[i], token[i+1]] += token_frequency_table[token]


    # 第4步开始训练BPE算法
    while len(vocab) < vocab_size: # 添加新的token直到词汇表达到指定大小
        if not pair_counts: # 如果没有数据可以处理了
            break

        # 找到频率最高的token对，为了通过测试，需要并处理平分情况
        max_count = max(pair_counts.values())
        # 找出所有频率最高的对，可能不止一个
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        # 在候选者中，选择字节序最大的那个
        best_pair = max(candidates)
        # 记录这次合并操作
        merges.append(best_pair)

        new_token_bytes = best_pair[0] + best_pair[1] # 将最佳token对的两个token连接起来

        # 将新token添加到词汇表，并记录这次合并操作
        vocab[current_next_id] = new_token_bytes
        current_next_id += 1 # 为下一个可能的新token准备ID
        
        #记录受影响的token，也就是包含best_pair的来自token_frequency_table的token
        affected_tokens = []
        for token, freq in token_frequency_table.items():
            has_pair = any(token[i:i+2] == best_pair for i in range(len(token) - 1))
            if has_pair:
                affected_tokens.append((token, freq))
        #从受影响的token中出发,每个token就是token_frequency_table的key
        for token, freq in affected_tokens:
            # 删除pair_counts中对应的best_pair
            for i in range(len(token) - 1):
                pair_counts[token[i], token[i+1]] -= freq
                if pair_counts[token[i], token[i+1]] <= 0:
                    del pair_counts[token[i], token[i+1]]
            # 将best_pair合并为new_token
            new_token_frequency_seq = merge_token_sequence(token, best_pair, new_token_bytes)
            # 更新pair_counts
            for i in range(len(new_token_frequency_seq)-1):
                pair = (new_token_frequency_seq[i], new_token_frequency_seq[i+1])
                pair_counts[pair] += freq
            # 更新token_frequency_table
            del token_frequency_table[token]
            token_frequency_table[new_token_frequency_seq] += freq

    # 保存词汇表到文件 (使用 pickle)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    # 保存合并操作记录到文件 (使用 pickle)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges # 返回最终的词汇表和合并记录

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab, merges = run_train_bpe("../data/owt_train.txt", 20000, [""])

# vocab, merges = run_train_bpe(data_path, vocab_size, special_tokens)
    print(vocab)
    print(merges)
'''
    
