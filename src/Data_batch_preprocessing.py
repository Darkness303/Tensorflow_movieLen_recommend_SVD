#coding:utf-8
from __future__ import absolute_import,division,print_function
import numpy as np
import pandas as pd
def read_data_process(filename,sep="\t"):
    col_names=["user","item","rate","st"]
    df=pd.read_csv(filename,sep=sep,header=None,names=col_names,engine="python")
    df["user"]-=1
    df["item"]-=1
    for col in ("user","item"):
        df[col]=df[col].astype(np.int32)

    df["rate"]=df["rate"].astype(np.float32)
    return df
class ShuffleDataIterator(object):
    """
    随机生成一个batch一个batch的数据

    """
    #初始化
    def __init__(self,inputs,batch_size=10):
        self.inputs=inputs
        self.batch_size=batch_size
        self.num_cols=len(self.inputs)
        self.len=len(self.inputs[0])
        self.inputs=np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self)]))
    #总样本量
    def _len_(self):
        return self
    def __iter__(self):
        return self
    #取出下一个batch
    def _next_(self):
        return self.next()
    def next(self):
        ids=np.random.randint(0,self.len,(self.batch_size))
        out=self.inputs[ids,:]
        return [out[:,i]for i in range(self.num_cols)]
class OneEpochDataIterator(ShuffleDataIterator):
    """
    顺序产出一个epoch的数据，在测试中可以能会用到
    """
    def __init__(self,inputs,batch_size=10):
        super(OneEpochDataIterator,self).__init__(inputs,batch_size=batch_size)
        if batch_size>0:
            self.idx_group=np.array_split(np.array(self.len),np.ceil(self.len/batch_size),)
        else:
            self.idx_group=[np.arange(self.len)]
    def next(self):
        if self.group_id>=len(self.idx_group):
            self.group_id=0
            raise  StopIteration
        out=self.inputs[self.idx_group[self.group_id],:]
        self.group_id+=1
        return [out[:,i]for i in range(self.num_cols)]