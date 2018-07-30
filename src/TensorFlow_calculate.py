#encoding:utf-8
import tensorflow as tf
#使用矩阵分解搭建的网络结构
def inference_svd(user_batch,item_batch,user_num,item_num,dim=5,device="/cpu:0"):
    # 使用CPU
    with tf.device("/cpu:0"):
        # 初始化几个bias
        global_bias=tf.get_variable("global_bias",shape=[])
        w_bias_user=tf.get_variable("embd_bias_user",shape=[user_num])
        w_bias_item=tf.get_variable("embd_bias_item",shape=[item_num])
        # bias向量
        bias_user=tf.nn.embedding_lookup(w_bias_user,user_batch,name="bias_user")
        bias_item=tf.nn.embedding_lookup(w_bias_item,item_batch,name="bias_item")
        w_user=tf.get_variable("embd_user",shape=[user_num,dim],
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item=tf.get_variable("embd_item",shape=[item_num,dim],
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        # user向量与item向量
        embd_user=tf.nn.embedding_lookup(w_user,user_batch,name="embedding_user")
        embd_item=tf.nn.embedding_lookup(w_item,item_batch,name="embedding_item")
        # 以上部分是TensorFlow初始化的部分 ，不管运行什么都需要的过程----------------
    with tf.device(device):
        # 按照实际的公式进行计算 先对user向量和item向量求内积
        infer=tf.reduce_sum(tf.multiply(embd_user,embd_item),1)
        # 加上几个偏置项
        infer=tf.add(infer,global_bias)
        infer=tf.add(infer,bias_user)
        infer=tf.add(infer,bias_item,name="svd_inference")
        # 加上正则项
        regularize=tf.add(tf.nn.l2_loss(embd_user),tf.nn.l2_loss(embd_item),name="svd_inference")
    return infer,regularize

def optimization(infer,regularize,rate_batch,learning_rate=0.001,reg=0.1,device="/cpu:0"):
    global_step=tf.train.get_global_step()
    assert global_step is not None
    # 选择合适的optimization做优化
    with tf.device(device):
        cost_l2=tf.nn.l2_loss(tf.subtract(infer,rate_batch))
        penalty=tf.constant(reg,dtype=tf.float32,shape=[],name="l2")
        cost=tf.add(cost_l2,tf.multiply(regularize,penalty))
        train_op=tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
    return cost,train_op
