import regex as re

def merge(pair,ID,byte_data):
    for piece in byte_data:
        for i in range(len(piece)-1):
            if (piece[i],piece[i+1])==pair:
                piece[i]=ID
                del piece[i+1]
            

def bpe_train(input_path,vocab_size,special_tokens):

    vocab = {i: bytes([i]) for i in range(256)}
    merges=[]
    pair_count={}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path,'r') as f:
        content=f.read()
        pieces=re.findall(PAT,content)
    byte_data=[list(piece.encode('utf-8')) for piece in pieces] #[[piece1],[piece2],...]，每一个最内的列表(piece)表示一个word, eg.['h','i']
    
    start_ID=256
    
    for i in special_tokens:
        vocab[start_ID]=i.encode('utf-8')
        start_ID+=1

    while len(vocab)<vocab_size:

        for piece in byte_data:
            for i in range(len(piece)-1):
                pair=(piece[i],piece[i+1])
                pair_count[pair]=pair_count.get(pair,0)+1 # {(pair):num,...}
        
        max_count=max(pair_count.values()) #最大频率
        max_pairs=[k for k,v in pair_count.items() if v==max_count] #一列有最大出现频率的pairs
        merge_pair=max(max_pairs) #选择最大字节序

        vocab[start_ID]=merge_pair[0]+merge_pair[1] #eg.['h','i'] -> ['hi']
        merges.append(merge_pair)

        merge(merge_pair,start_ID,byte_data) #更新byte_data，一对转化为一个

        start_ID+=1
        pair_count={}
    
    return vocab,merges


    