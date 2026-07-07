import pickle

class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens
        self.byte_to_id={v:k for k,v in self.vocab.items()}
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

        byte_txt=[bytes([x]) for x in list(text.encode())]
        id_txt=[self.byte_to_id[byte_txt[i:i+1]] for i in range(byte_txt)]
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
    



                
                
        
