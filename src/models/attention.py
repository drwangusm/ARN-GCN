import keras
import keras.backend as K
import tensorflow as tf
import tensorflow.nn as nn
import numpy as np

#rewrite of attention model
#
#    输入序列 X (N, T, D)
#        |
#        v
#   +-----------+
#   |   投影    |  W1
#   +-----------+
#        |
#        v
#    激活函数 tanh
#        |
#        v
#    +-----------+
#    |   投影    |  W2
#    +-----------+
#        |
#        v
#    softmax 计算注意力权重
#        |
#        v
#    注意力权重 a (N, T, 1)
#        |
#        v
#    输入序列 X (N, T, D) * 注意力权重 a (N, T, 1)
#        |
#        v
#    +----------------------+
#    |  加权后的输入序列    |  output
#    +----------------------+
#        |
#        v
#    时间步维度求和
#        |
#        v
#    加权后的输出 sentence (N, D)

class IRNAttention(keras.layers.Layer):
    def __init__(self, num_head=1, projection_size=None, return_attention=False, **kwargs):
        self.return_attention = return_attention
        self.projection_size = projection_size
        self.num_head = num_head
        super(IRNAttention, self).__init__(**kwargs)

    def build(self, input_size):
        if self.projection_size is None:
            self.projection_size = input_size[0][1] # size of joint object

        #todo add another layer
        #todo add bias

        self.w1=self.add_weight(name="Att1", shape=(input_size[0][1], self.projection_size), initializer="normal") #name property is useful for avoiding RuntimeError: Unable to create link.
        self.w2=self.add_weight(name="Att2", shape=(self.projection_size, 1), initializer="normal")

        super(IRNAttention, self).build(input_size)

    def call(self, inputs):

        inputs = tf.transpose(tf.stack(inputs), perm=[1,0,2])
        e = K.tanh(K.dot(inputs,self.w1))
        e2 = K.dot(e, self.w2)
        a = K.softmax(e2, axis=1)
        output = inputs*a
        ##return the ouputs. 'a' is the set of attention weights
        ##the second variable is the 'attention adjusted o/p state' or context
        sentence = K.sum(output, axis=1)
        attention = a
        if self.return_attention:
            return [sentence, attention]
        else:
            return (sentence)

    def get_config(self):
        config = super(IRNAttention, self).get_config()
        config.update({"num_head": self.num_head,
                       "projection_size": self.projection_size,
                       "return_attention": self.return_attention})
        return config
    #增加实现
    def compute_output_shape(self, input_shape):
        # print("input_shape:",input_shape)
        batch_size = input_shape[0][0]  # 获取批量大小
        print("batch_size:",batch_size)
         # 检查输入形状是否正确
        if len(input_shape) < 2:
            raise ValueError("Input shape must be a tuple of length at least 2")
        features = input_shape[0][1]  # 获取每个对象的特征数
        if self.return_attention:
            # 如果返回了注意力权重，输出形状应为 (batch_size, features), (batch_size, num_objs, 1)
            return [(batch_size, features), (batch_size, input_shape[0][1], 1)]
        else:
            # 如果没有返回注意力权重，输出形状应为 (batch_size, features)
            return (batch_size, features)





