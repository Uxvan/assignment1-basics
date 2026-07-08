'''
import pickle
import regex as re
import re as builtin_re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens or []
        
        self.byte_to_id={v:k for k,v in self.vocab.items()}
        self.sorted_sp_toks=sorted(self.special_tokens,key=len,reverse=True)#当special_tokens里元素有包含关系，如果按照给定顺序可能会让短的先匹配，长的失去机会eg."<|endoftext|>" 和 "<|endoftext|><|endoftext|>" 
        self.sp_tok_id={v:self.byte_to_id[v.encode()] for v in self.sorted_sp_toks}
        self.merges_to_id=[(self.byte_to_id[x],self.byte_to_id[y]) for x,y in merges]

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath, special_tokens=None):
        with open(vocab_filepath,'rb') as f:
            vocab=pickle.load(f)
        with open(merges_filepath,'rb') as f:
            merges=pickle.load(f)
        special_tokens=special_tokens
        return cls(vocab,merges,special_tokens)

    def encode(self,text:str)->list[int]:

        if self.sorted_sp_toks:
            special_pattern='|'.join(builtin_re.escape(tok) for tok in self.sorted_sp_toks) 
            parts=builtin_re.split(f'({special_pattern})',text) #把文本按照special_tokens分割为大块,同时special_tokens作为独立元素保留在结果
        else:#如果special_tokens为空，special_pattern 会是空字符串 ''，这时 builtin_re.split('()', text) 会在每个字符之间都切一刀，所以最好跳过分割
            parts=[text]
        
        all_id_to_txt=[]
        for p in parts:
            if p in self.special_tokens:
                all_id_to_txt.append(self.sp_tok_id[p])
            else:
                pieces=re.findall(PAT,p)
                for piece in pieces:
                    id_txt=[]
                    id_txt.extend([self.byte_to_id[bytes([x])] for x in piece.encode()])
                    id_pairs=list(zip(id_txt[:-1],id_txt[1:]))

                    #按merges优先级找应该被合并的项
                    j=0
                    while j<len(self.merges):
                        goal_id_pair=self.merges_to_id[j]
                        goal_merged_byte=self.vocab[goal_id_pair[0]]+self.vocab[goal_id_pair[1]]
                        new_id=self.byte_to_id[goal_merged_byte]
                        i=0
                        while i<len(id_pairs):
                            if id_pairs[i]==goal_id_pair:
                                id_txt[i]=new_id
                                if i>0:
                                    id_pairs[i-1]=(id_txt[i-1],new_id)
                                if i<len(id_pairs)-1:
                                    id_pairs[i+1]=(new_id,id_txt[i+2])
                                del id_pairs[i]
                                del id_txt[i+1]
                            else:
                                i+=1
                        j+=1
                    all_id_to_txt.extend(id_txt)
        return all_id_to_txt
        
        
    def encode_iterable(self,iterable):
       for text in iterable:
           yield from self.encode(text)

    def decode(self, ids:list[int]) -> str:
        bytes=[]
        for i in ids:
            bytes.append(self.vocab[i])
        result=b''.join(bytes)
        text=result.decode('utf-8',errors='ignore')
        return text
'''
import pickle
import regex as re
import re as builtin_re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens or []
        
        self.byte_to_id={v:k for k,v in self.vocab.items()}

        self.sorted_sp_toks=sorted(self.special_tokens,key=len,reverse=True)#当special_tokens里元素有包含关系，如果按照给定顺序可能会让短的先匹配，长的失去机会eg."<|endoftext|>" 和 "<|endoftext|><|endoftext|>" 
        self.sp_tok_id={v:self.byte_to_id[v.encode()] for v in self.sorted_sp_toks}

        self.merges_to_priority={self.merges[i]:i for i in range(len(self.merges))}

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath, special_tokens=None):
        with open(vocab_filepath,'rb') as f:
            vocab=pickle.load(f)
        with open(merges_filepath,'rb') as f:
            merges=pickle.load(f)
        special_tokens=special_tokens
        return cls(vocab,merges,special_tokens)

    def encode(self,text:str)->list[int]:
        if self.sorted_sp_toks:
            special_pattern='|'.join(builtin_re.escape(tok) for tok in self.sorted_sp_toks) 
            parts=builtin_re.split(f'({special_pattern})',text) #把文本按照special_tokens分割为大块,同时special_tokens作为独立元素保留在结果
        else:#如果special_tokens为空，special_pattern 会是空字符串 ''，这时 builtin_re.split('()', text) 会在每个字符之间都切一刀，所以最好跳过分割
            parts=[text]
        
        all_id_to_txt=[]
        for p in parts:
            if p in self.special_tokens:
                all_id_to_txt.append(self.sp_tok_id[p])
            else:
                pieces=re.findall(PAT,p)
                for piece in pieces:
                    piece_id=[]
                    piece_bytes=piece.encode()
                    piece_id.extend([self.byte_to_id[bytes([x])] for x in piece_bytes])

                    #得到piece字节对的id和优先级
                    pairs_id=list(zip(piece_id[:-1],piece_id[1:]))
                    pairs_priority=[self.merges_to_priority.get((self.vocab[pd[0]],self.vocab[pd[1]]),-1) for pd in pairs_id]#-1表示该字节对不在self.merges

                    #按merges优先级找应该被合并的项
                    #在encode阶段，merges数量级远大于text，所以要遍历的是piece，这与训练bpe时相反
                    while True:

                        #得到每一轮优先合并的字节对在pairs_priority的索引'min_prior_index'，
                        #等于该字节对在pairs_id的索引
                        #同时也是该字节对的第一位字节在piece_bytes中的索引及其id在piece_id中索引
                        min_prior=min([x for x in pairs_priority if x>-1],default=-2)#default=-2防止列表为空时error
                        
                        if min_prior>=0: #仍然存在需要合并的字节对
                            
                            min_prior_index=pairs_priority.index(min_prior)
                            min_prior_pair_id=pairs_id[min_prior_index]
                            merged_byte=self.vocab[min_prior_pair_id[0]]+self.vocab[min_prior_pair_id[0]][1] #合并结果
                            merged_id=self.byte_to_id[merged_byte]#合并结果对应id

                            #修改piece_id和pairs_priority
                            piece_id[min_prior_index]=merged_id
                            del piece_id[min_prior_index+1]

                            if min_prior_index>0:
                                pairs_priority[min_prior_index-1]=self.merges_to_priority.get((piece_bytes[min_prior_index-1:merged_byte],merged_byte),-1)
                            if min_prior_index<len(pairs_priority)-1:
                                pairs_priority[min_prior_index+1]=self.merges_to_priority.get((merged_byte,piece_bytes[min_prior_index+2:merged_byte+3]),-1)
                            del pairs_priority[min_prior_index]

                        else:
                            break

                    all_id_to_txt.extend(piece_id)
        return all_id_to_txt
        
        
    def encode_iterable(self,iterable):
       for text in iterable:
           yield from self.encode(text)

    def decode(self, ids:list[int]) -> str:
        bytes=[]
        for i in ids:
            bytes.append(self.vocab[i])
        result=b''.join(bytes)
        text=result.decode('utf-8',errors='ignore')
        return text
    



                
                
        



                
                
        



                
                
        
