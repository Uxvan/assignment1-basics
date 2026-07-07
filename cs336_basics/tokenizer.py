import pickle

class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath, special_tokens=None):
        with open(vocab_filepath,'rb') as f:
            vocab=pickle.load(f)
        with open(merges_filepath,'rb') as f:
            merges=pickle.load(f)
        special_tokens=special_tokens
        return cls(vocab,merges,special_tokens)

    def encode(self,text:str)->list[int]:
        b_txt=text.encode()
        int_txt=list(b_txt)
        int_pairs=list(zip(int_txt[:-1],int_txt[1:]))

        #找应该被合并的项
        i=0
        while i<len(int_pairs):
            p=int_pairs[i]
            byte_p=(self.vocab[p[0]],self.vocab[p[1]])
            merged_bytes=bytes(p)
            if byte_p in self.merges:
                num=next(k for k,v in self.vocab.items() if v==merged_bytes)
                int_txt[i]=num
                del int_txt[i+1]
                if i>0:
                    int_pairs[i-1][1]=num
                if i<len(int_pairs)-1:
                    int_pairs[i+1][0]=num
                del int_pairs[i]
            else:
                i+=1
        return int_txt
    
    def encode_iterable(self,iterable):
       for text in iterable:
           yield from self.encode(text)

    def decode(self, ids:list[int]) -> str:
        bytes=[]
        for i in ids:
            bytes.append(self.vocab[i])
        result=b''.join(bytes)
        text=result.decode('utf-8')
        return text
    



                
                
        
