import time
from collections import deque
import sys
import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2
from src.TensorFlow_calculate import inference_svd,optimization
from src.Data_batch_preprocessing import read_data_and_process,ShuffleDataIterator,OneEpochDataIterator
np.random.seed(13575)
# 一批数据的大小
Batch_size=1000
# 用户数据
User_Num=6040
# 电影数据
Item_Num=3952
# factor纬度
DIM=15
# 最大迭代的轮数
EPOCH_Max=200
# 使用cpu做训练
Device="/cpu:0"
#截断
def clip(x):
    return  np.clip(x,1.0,5.0)
# 这个是方便TensorFlow可视化做的summary
def make_scalar_summary(name,val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name,simple_value=val)])
# 调用上面的函数截获数据
def get_data():
    df=read_data_and_process(r"E:\JiangIntellijWorkingSpace\tools\Tensor_movieLen_SVD\ml-1m\ratings.dat",sep="::")
    rows=len(df)
    df=df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index=int(rows*0.9)
    df_train=df[0:split_index]
    df_test=df[split_index:].reset_index(drop=True)
    print (df_train.shape,df_test.shape)
    return df_train,df_test
# 实际训练过程
def svd(train,test):
    samples_per_batch=len(train)//Batch_size
    # 一批一批数据用户训练
    iter_train=ShuffleDataIterator([train["user"],train["item"],train["rate"]],batch_size=Batch_size)
    # 测试数据
    iter_test=OneEpochDataIterator([test["user"],test["item"],test["rate"]],batch_size=-1)
    # user 和item batch
    user_batch=tf.placeholder(tf.int32,shape=[None],name="id_user")
    item_batch=tf.placeholder(tf.int32,shape=[None],name="id_item")
    rate_batch=tf.placeholder(tf.float32,shape=[None])
    # 构建graph和训练
    infer,regularizer=inference_svd(user_batch,item_batch,user_num=User_Num,item_num=Item_Num,device=Device)
    global_step=tf.contrib.framework.get_or_create_global_step()
    _,train_op=optimization(infer,regularizer,rate_batch,learning_rate=0.001,reg=0.05)
    # 初始化所有的变量
    init_op=tf.global_variables_initializer()
    # 开始迭代
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer=tf.summary.FileWriter(logdir=r"E:\JiangIntellijWorkingSpace\tools\Tensor_movieLen_SVD\log",graph=sess.graph)
        print("{} {} {} {}".format("epoch","train_error","val_error","elappsed_time"))
        errors=deque(maxlen=samples_per_batch)
        start=time.time()
        for i in range(EPOCH_Max * samples_per_batch):
            user,item,rate=next(iter_train)
            _,pred_batch=sess.run([train_op,infer],feed_dict={user_batch:user,item_batch:item,rate_batch:rate})
            pred_batch=clip(pred_batch)
            errors.append(np.power(pred_batch-rate,2))
            # print ("i=",i,"sample_per_batch=",samples_per_batch,"i%sample=",i %samples_per_batch)
            if (i%samples_per_batch)==0:
                train_err=np.sqrt(np.mean(errors))
                test_err2=np.array([])
                for users,items,rates in iter_test:
                    pred_batch=sess.run(infer,feed_dict={user_batch:users,item_batch:items})
                    pred_batch=clip(pred_batch)
                    test_err2=np.append(test_err2,np.power(pred_batch-rates,2))
                end=time.time()
                test_err=np.sqrt(np.mean(test_err2))
                print("{:7d} {:f} {:f} {:f}(s)".format(i//samples_per_batch,train_err,test_err,end-start))

                train_err_summary=make_scalar_summary("training_error",train_err)
                test_err_summary=make_scalar_summary("test-error",test_err)
                summary_writer.add_summary(train_err_summary,i)
                summary_writer.add_summary(test_err_summary,i)
                start=end
df_train,df_test=get_data()
svd(df_train,df_test)