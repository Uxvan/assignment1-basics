import regex as re
import re as builtin_re
from collections import defaultdict

def train_bpe(input_path,vocab_size,special_tokens):

    vocab = {i: bytes([i]) for i in range(256)}
    merges=[]

    next_ID=256
    for tok in special_tokens:
        tok_bytes=tok.encode("utf-8")
        if tok_bytes not in set(vocab.values()):
            vocab[next_ID]=tok_bytes
            next_ID+=1

    #一、分割文本，每个word，标点等转为bytes，并统计频率
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_pattern='|'.join(builtin_re.escape(tok) for tok in special_tokens) 

    with open(input_path,'r') as f:
        content=f.read()

    parts=builtin_re.split(f'{special_pattern}',content) #把文本按照special_tokens分割为大块,同时special_tokens作为独立元素保留在结果
    normal_parts=[p for p in parts if p not in special_tokens] 

    #在大分割里小分割，按照空格和标点，由每一大块得到['hello','world',...]
    pieces=[]
    for p in normal_parts:
        piece=re.findall(PAT,p) 
        pieces.extend(piece) #pieces中元素为word
    
    token_freq=defaultdict(int)
    for word in pieces: 
        word_bytes=word.encode('utf-8')
        bytes_list=[bytes([x]) for x in word_bytes]#[b'h',b'e',b'l',b'l',b'o']
        token_freq[tuple(bytes_list)]+=1
        
    
    #二、初始化字节对
    pair_count=defaultdict(int)
    for piece in token_freq.keys():
        for i in range(len(piece)-1):
            pair=(piece[i],piece[i+1])
            pair_count[pair]+=token_freq[piece] # {(pair):count,...}


    #三、合并循环
    while len(vocab)<vocab_size:
        if not pair_count:
            break

        max_count=max(pair_count.values()) #最大频率
        max_pairs=[k for k,v in pair_count.items() if v==max_count] #一列有最大出现频率的pairs
        merge_pair=max(max_pairs) #选择最大字节序

        vocab[next_ID]=merge_pair[0]+merge_pair[1] #eg.['h','i'] -> ['hi'],注意要先把token转换为字符
        merges.append(merge_pair)

        #更新pair_count
        new_token_freq=defaultdict(int)
        pair_count.pop(merge_pair)
        for tok,freq in token_freq.items():
            i=0
            new_tok=[]
            while i<len(piece)-1:
                if (tok[i],tok[i+1])==merge_pair:
                    if i+2<len(tok):#减去右邻pair,加上新右邻
                        pair_count[(tok[i+1],tok[i+2])]-=freq #以merge_pair=ab为例，减去[d,a,b,c]中的pair(b,c)个数
                        pair_count[(vocab[next_ID],piece[i+2])]+=freq #添加(ab,c)
                    if  i>0:#减去旧左邻,加上新左邻
                        pair_count[(tok[i-1],tok[i])]-=freq
                        pair_count[(tok[i-1],vocab[next_ID])]=freq

                    new_tok.append(vocab[next_ID])
                    i+=2
                    
                else:
                    new_tok.append(tok[i])
                    i+=1
            new_token_freq[tuple(new_tok)]+=freq
        token_freq=new_token_freq
        next_ID+=1
    
    return vocab,merges


    
