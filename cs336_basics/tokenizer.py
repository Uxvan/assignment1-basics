'''
import pickle
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

        special_pattern='|'.join(builtin_re.escape(tok) for tok in self.sorted_sp_toks) 
        parts=builtin_re.split(f'({special_pattern})',text) #把文本按照special_tokens分割为大块,同时special_tokens作为独立元素保留在结果
        id_txt=[]
        for p in parts:
            if p in self.special_tokens:
                id_txt.append(self.sp_tok_id[p])
            else:
                id_txt.extend([self.byte_to_id[bytes([x])] for x in p.encode()])

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
        return id_txt
        
        
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
            parts=text
        
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
    



                
                
        



                
                
        
