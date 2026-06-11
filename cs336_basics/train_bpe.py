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
                    if i+2<len(piece):#减去右邻pair
                        pair_count[(piece[i+1],piece[i+2])]-=1 #以merge_pair=ab为例，减去[d,a,b,c]中的pair(b,c)个数
                    if  i>0:#减去旧左邻
                        pair_count[(piece[i-1],piece[i])]-=1
                        
                    #更新byte_data
                    piece[i]=start_ID
                    del piece[i+1]

                    if i+2<len(piece):#加上新右邻
                        pair_count[(start_ID,piece[i+1])]=pair_count.get((start_ID,piece[i+1]),0)+1 #添加(ab,c)
                    if i>0:#加上新左邻
                        pair_count[(piece[i-1],start_ID)]=pair_count.get((piece[i-1],start_ID),0)+1
                i+=1
        start_ID+=1
    
    return vocab,merges


    
