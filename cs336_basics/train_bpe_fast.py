import regex as re
import re as builtin_re
from collections import defaultdict, Counter
import pickle
import heapq
from multiprocessing import Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class NegPair:
    """包装pair,让heapq(默认最小堆)按 (count最大, pair字节序最大) 弹出。"""
    __slots__=("pair",)
    def __init__(self,pair):
        self.pair=pair
    def __lt__(self,other):
        return self.pair>other.pair  # 反转比较,tie时字节序大的先出堆
    def __eq__(self,other):
        return self.pair==other.pair

def _count_words_in_parts(parts):
    """子进程worker:对一批text part做GPT-2风格预分词,返回{token_tuple: count}"""
    local=Counter()
    for p in parts:
        for word in re.findall(PAT,p):
            word_bytes=word.encode('utf-8')
            local[tuple(bytes([x]) for x in word_bytes)]+=1
    return local

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
    special_pattern='|'.join(builtin_re.escape(tok) for tok in special_tokens) 

    with open(input_path,'r') as f:
        content=f.read()

    parts=builtin_re.split(f'{special_pattern}',content) #把文本按照special_tokens分割为大块,同时special_tokens作为独立元素保留在结果
    normal_parts=[p for p in parts if p not in special_tokens] 

    #用多进程做预分词(GPT-2 pattern的regex.findall在大文本上很慢,是单进程主要瓶颈之一)
    n_proc=max(1,cpu_count()-1)
    chunk_size=max(1,len(normal_parts)//n_proc)
    chunks=[normal_parts[i:i+chunk_size] for i in range(0,len(normal_parts),chunk_size)]

    token_freq=defaultdict(int) #{(token):int,...}
    if len(chunks)>1:
        with Pool(processes=n_proc) as pool:
            for local_counter in pool.map(_count_words_in_parts,chunks):
                for tok,c in local_counter.items():
                    token_freq[tok]+=c
    else:
        for tok,c in _count_words_in_parts(normal_parts).items():
            token_freq[tok]+=c

    
    #二、初始化pair_count:{(pair):count,...}; pair_token:{(pair):(token1,...)}, 哪些token包含某个pair
    pair_count=defaultdict(int)
    pair_token=defaultdict(set) #对于token[a,a,a]，set保证对于pair(a,a)，不会出现两次该token
    for tok in token_freq.keys():
        for i in range(len(tok)-1):
            pair=(tok[i],tok[i+1])
            pair_count[pair]+=token_freq[tok] 
            pair_token[pair].add(tok)

    heap=[(-c,NegPair(p)) for p,c in pair_count.items()]
    heapq.heapify(heap) #堆顶始终是(频率最大,tie时字节序最大)的pair

    #三、合并循环
    while len(vocab)<vocab_size:
        #从堆顶取出当前仍然有效(未被后续更新弄脏)的最大pair,懒删除过期条目
        merge_pair=None
        while heap:
            neg_count,np=heapq.heappop(heap)
            pair=np.pair
            cur=pair_count.get(pair,0)
            if cur==-neg_count and cur>0: #堆里的这条记录和当前真实计数一致,才是有效的
                merge_pair=pair
                break
            #否则是过期条目(计数已变),直接丢弃继续弹
        if merge_pair is None:
            break

        vocab[next_ID]=merge_pair[0]+merge_pair[1] #eg.['h','i'] -> ['hi'],注意要先把token转换为字符
        merges.append(merge_pair)
        merge_bytes=vocab[next_ID]

        #更新pair_count, token_freq, pair_token.                       
        pair_count.pop(merge_pair)
        affected_pairs=list(pair_token.pop(merge_pair))
        touched=set() #本次merge中被改动过计数的pair,处理完affected_pairs后统一push进堆
        for tok in affected_pairs:
            freq=token_freq[tok]
            i=0
            new_tok=[] #bytes合并后得到的tok

            while i<len(tok):
                if i<len(tok)-1 and (tok[i],tok[i+1])==merge_pair :
                    
                    if i+2<len(tok):
                        #更新pair_count减去右邻pair,加上新右邻
                        pair_count[(tok[i+1],tok[i+2])]-=freq #以merge_pair=ab为例，减去[d,a,b,c]中的pair(b,c)个数
                        pair_count[(merge_bytes,tok[i+2])]+=freq #添加(ab,c)
                        #pair_token引入新pair与对应token
                        pair_token[(merge_bytes,tok[i+2])].add(tok)
                        touched.add((tok[i+1],tok[i+2]))
                        touched.add((merge_bytes,tok[i+2]))
                        
                    if  i>0:
                        #更新pair_count减去旧左邻,加上新左邻
                        pair_count[(tok[i-1],tok[i])]-=freq
                        pair_count[(tok[i-1],merge_bytes)]+=freq
                        #pair_token引入新pair与对应token
                        pair_token[(tok[i-1],merge_bytes)].add(tok)
                        touched.add((tok[i-1],tok[i]))
                        touched.add((tok[i-1],merge_bytes))

                    new_tok.append(vocab[next_ID])
                    i+=2
                    
                else:
                    new_tok.append(tok[i])
                    i+=1
            new_tok=tuple(new_tok)
            f=token_freq.pop(tok)
            token_freq[new_tok]=f
        
            #把pair_token里所有pair对应的‘tok’换为new_tok, 只需看new_tok对应哪些pair (pair_token中merge_pair已删除，并加入了新的pair)
            for i in range(len(new_tok)-1):
                p=(new_tok[i],new_tok[i+1])
                pair_token[p].discard(tok)
                pair_token[p].add(new_tok)

        #把这一轮真正变动过的pair按最新计数推入堆;计数归零的顺手从pair_count里清掉,防止字典无限膨胀
        for p in touched:
            c=pair_count.get(p,0)
            if c>0:
                heapq.heappush(heap,(-c,NegPair(p)))
            elif p in pair_count:
                pair_count.pop(p)

        next_ID+=1
    
    #写为文件
    with open('vocab.pkl','wb') as f:
        pickle.dump(vocab,f)
    
    with open('merges.pkl','wb') as f:
        pickle.dump(merges,f)
        
    return vocab,merges