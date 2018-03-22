import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys


size=256
batch_size=10
num_batch=100
img=[]
label=[]
img_batch=[]
label_batch=[]
n_classes=120
Epoch = 10
learn_rate = 0.01

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [size, size, 3])
    img = tf.cast(img, tf.float32)*(1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    
    #print(label)

    return img, label


def load_data():

    img,label=read_and_decode("../tfrecord/vaild.tfrecords")


    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size, capacity=2000,
                                                    min_after_dequeue=1000)
    label_batch = tf.one_hot(label_batch, depth= n_classes)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    return img_batch,label_batch
    #print(label_batch)

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, n_classes])

def inception_unit(inputdata, weights, biases):
    # A3 inception 3a
    inception_in = inputdata

    # Conv 1x1+S1
    inception_1x1_S1 = tf.nn.conv2d(inception_in, weights['inception_1x1_S1'], strides=[1,1,1,1], padding='SAME')
    inception_1x1_S1 = tf.nn.bias_add(inception_1x1_S1, biases['inception_1x1_S1'])
    inception_1x1_S1 = tf.nn.relu(inception_1x1_S1)
    # Conv 3x3+S1
    inception_3x3_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_3x3_S1_reduce'], strides=[1,1,1,1], padding='SAME')
    inception_3x3_S1_reduce = tf.nn.bias_add(inception_3x3_S1_reduce, biases['inception_3x3_S1_reduce'])
    inception_3x3_S1_reduce = tf.nn.relu(inception_3x3_S1_reduce)
    inception_3x3_S1 = tf.nn.conv2d(inception_3x3_S1_reduce, weights['inception_3x3_S1'], strides=[1,1,1,1], padding='SAME')
    inception_3x3_S1 = tf.nn.bias_add(inception_3x3_S1, biases['inception_3x3_S1'])
    inception_3x3_S1 = tf.nn.relu(inception_3x3_S1)
    # Conv 5x5+S1
    inception_5x5_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_5x5_S1_reduce'], strides=[1,1,1,1], padding='SAME')
    inception_5x5_S1_reduce = tf.nn.bias_add(inception_5x5_S1_reduce, biases['inception_5x5_S1_reduce'])
    inception_5x5_S1_reduce = tf.nn.relu(inception_5x5_S1_reduce)
    inception_5x5_S1 = tf.nn.conv2d(inception_5x5_S1_reduce, weights['inception_5x5_S1'], strides=[1,1,1,1], padding='SAME')
    inception_5x5_S1 = tf.nn.bias_add(inception_5x5_S1, biases['inception_5x5_S1'])
    inception_5x5_S1 = tf.nn.relu(inception_5x5_S1)
    # MaxPool
    inception_MaxPool = tf.nn.max_pool(inception_in, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    inception_MaxPool = tf.nn.conv2d(inception_MaxPool, weights['inception_MaxPool'], strides=[1,1,1,1], padding='SAME')
    inception_MaxPool = tf.nn.bias_add(inception_MaxPool, biases['inception_MaxPool'])
    inception_MaxPool = tf.nn.relu(inception_MaxPool)
    # Concat
    #tf.concat(concat_dim, values, name='concat')
    #concat_dim 是 tensor 连接的方向（维度）， values 是要连接的 tensor 链表， name 是操作名。 cancat_dim 维度可以不一样，其他维度的尺寸必须一样。
    inception_out = tf.concat([inception_1x1_S1, inception_3x3_S1, inception_5x5_S1, inception_MaxPool],3)
    return inception_out

def GoogleLeNet_topological_structure(x):
    # A0 输入数据
    x = tf.reshape(x,[-1,256,256,3])  # 调整输入数据维度格式

    # A1  Conv 7x7_S2
    x = tf.nn.conv2d(x, weights['conv1_7x7_S2'], strides=[1,2,2,1], padding='SAME')
    # 卷积层 卷积核 7*7 扫描步长 2*2 [128,128,64]
    x = tf.nn.bias_add(x, biases['conv1_7x7_S2'])
    #print (x.get_shape().as_list())
    # 偏置向量
    x = tf.nn.relu(x)
    # 激活函数
    x = tf.nn.max_pool(x, ksize=pooling['pool1_3x3_S2'], strides=[1,2,2,1], padding='SAME')
    # 池化取最大值[64,64,64]
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    # 局部响应归一化
    temp_C1R1 = x
    #res-nets
    x = tf.nn.conv2d(x, weights['conv1_3x3_R1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv1_3x3_R1'])

    x = tf.concat([x, temp_C1R1],3)
    x = tf.nn.conv2d(x, weights['conv1_1x1_R1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv1_1x1_R1'])
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    
    x = tf.nn.max_pool(x, ksize=pooling['pool1_3x3_R1'], strides=[1,1,1,1], padding='SAME' )    
    temp_C1R2 = x
    #[64,64,64]

    x = tf.nn.conv2d(x, weights['conv1_3x3_R2'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv1_3x3_R2'])

    x = tf.concat([x, temp_C1R2],3)
    x = tf.nn.conv2d(x, weights['conv1_1x1_R2'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv1_1x1_R2'])
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    x = tf.nn.max_pool(x, ksize=pooling['pool1_3x3_R2'], strides=[1,1,1,1], padding='SAME' ) 
   
    # A2
    x = tf.nn.conv2d(x, weights['conv2_1x1_S1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_1x1_S1'])
    #[64,64,64]
    x = tf.nn.conv2d(x, weights['conv2_3x3_S1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_3x3_S1'])
    #[64,64,192]
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    x = tf.nn.max_pool(x, ksize=pooling['pool2_3x3_S2'], strides=[1,2,2,1], padding='SAME')
    #[32,32,192]

    # inception 3
    inception_3a = inception_unit(inputdata=x, weights=conv_W_3a, biases=conv_B_3a)
    #[32,32,256]
    inception_3b = inception_unit(inception_3a, weights=conv_W_3b, biases=conv_B_3b)
    #[32,32,480]

    # 池化层
    x = inception_3b
    x = tf.nn.max_pool(x, ksize=pooling['pool3_3x3_S2'], strides=[1,2,2,1], padding='SAME' )
    temp_C2R1 = x
    #[16,16,480]

    #res-nets
    x = tf.nn.conv2d(x, weights['conv2_3x3_R1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_3x3_R1'])

    x = tf.concat([x, temp_C2R1],3)
    x = tf.nn.conv2d(x, weights['conv2_1x1_R1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_1x1_R1'])
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    
    x = tf.nn.max_pool(x, ksize=pooling['pool2_3x3_R1'], strides=[1,1,1,1], padding='SAME' )    
    temp_C2R2 = x
    #[16,16,480]

    x = tf.nn.conv2d(x, weights['conv2_3x3_R2'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_3x3_R2'])

    x = tf.concat([x, temp_C2R2],3)
    x = tf.nn.conv2d(x, weights['conv2_1x1_R2'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_1x1_R2'])
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    x = tf.nn.max_pool(x, ksize=pooling['pool2_3x3_R2'], strides=[1,1,1,1], padding='SAME' )    

    # inception 4
    inception_4a = inception_unit(inputdata=x, weights=conv_W_4a, biases=conv_B_4a)
    #[16,16,512]
    # 引出第一条分支
    #softmax0 = inception_4a
    inception_4b = inception_unit(inception_4a, weights=conv_W_4b, biases=conv_B_4b)
    #[16,16,512]    
    inception_4c = inception_unit(inception_4b, weights=conv_W_4c, biases=conv_B_4c)
    #[16,16,512]
    inception_4d = inception_unit(inception_4a, weights=conv_W_4d, biases=conv_B_4d)
    #[16,16,528]

    # 引出第二条分支
    #softmax1 = inception_4d
    inception_4e = inception_unit(inception_4d, weights=conv_W_4e, biases=conv_B_4e)
    #[16,16,832]

    # 池化
    x = inception_4e
    x = tf.nn.max_pool(x, ksize=pooling['pool4_3x3_S2'], strides=[1,2,2,1], padding='SAME' )
    #[8,8,832]

    # inception 5
    inception_5a = inception_unit(x, weights=conv_W_5a, biases=conv_B_5a)
    #[8,8,832]
    inception_5b = inception_unit(inception_5a, weights=conv_W_5b, biases=conv_B_5b)
    #[8,8,1024]
    softmax2 = inception_5b

    # 后连接
    softmax2 = tf.nn.avg_pool(softmax2, ksize=[1,8,8,1], strides=[1,1,1,1], padding='SAME')
    softmax2 = tf.nn.dropout(softmax2, keep_prob=0.4)
    softmax2 = tf.reshape(softmax2, [-1,weights['FC2'].get_shape().as_list()[0]])
    softmax2 = tf.nn.bias_add(tf.matmul(softmax2,weights['FC2']),biases['FC2'])
    #print(softmax2.get_shape().as_list())
    return softmax2

weights = {
    'conv1_7x7_S2': tf.Variable(tf.random_normal([7,7,3,64])),
    'conv2_1x1_S1': tf.Variable(tf.random_normal([1,1,64,64])),
    'conv2_3x3_S1': tf.Variable(tf.random_normal([3,3,64,192])),
    'conv1_1x1_R1': tf.Variable(tf.random_normal([1,1,192,64])),
    'conv1_3x3_R1': tf.Variable(tf.random_normal([3,3,64,128])),
    'conv1_1x1_R2': tf.Variable(tf.random_normal([1,1,192,64])),
    'conv1_3x3_R2': tf.Variable(tf.random_normal([3,3,64,128])),
    'conv2_1x1_R1': tf.Variable(tf.random_normal([1,1,680,480])),
    'conv2_3x3_R1': tf.Variable(tf.random_normal([3,3,480,200])),
    'conv2_1x1_R2': tf.Variable(tf.random_normal([1,1,680,480])),
    'conv2_3x3_R2': tf.Variable(tf.random_normal([3,3,480,200])),
    'FC2': tf.Variable(tf.random_normal([8*8*1024, 120]))
}

biases = {
    'conv1_7x7_S2': tf.Variable(tf.random_normal([64])),
    'conv2_1x1_S1': tf.Variable(tf.random_normal([64])),
    'conv2_3x3_S1': tf.Variable(tf.random_normal([192])),
    'conv1_1x1_R1': tf.Variable(tf.random_normal([64])),
    'conv1_3x3_R1': tf.Variable(tf.random_normal([128])),
    'conv1_1x1_R2': tf.Variable(tf.random_normal([64])),
    'conv1_3x3_R2': tf.Variable(tf.random_normal([128])),
    'conv2_1x1_R1': tf.Variable(tf.random_normal([480])),
    'conv2_3x3_R1': tf.Variable(tf.random_normal([200])),
    'conv2_1x1_R2': tf.Variable(tf.random_normal([480])),
    'conv2_3x3_R2': tf.Variable(tf.random_normal([200])),
    'FC2': tf.Variable(tf.random_normal([120]))

}
pooling = {
    'pool1_3x3_S2': [1,3,3,1],
    'pool2_3x3_S2': [1,3,3,1],
    'pool3_3x3_S2': [1,3,3,1],
    'pool4_3x3_S2': [1,3,3,1],
    'pool1_3x3_R1': [1,3,3,1],
    'pool1_3x3_R2': [1,3,3,1],
    'pool2_3x3_R1': [1,3,3,1],
    'pool2_3x3_R2': [1,3,3,1]
}
conv_W_3a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,192,64])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,192,96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,96,128])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,192,16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,16,32])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,192,32]))

}
conv_B_3a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([64])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([128])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([32])),
    'inception_MaxPool': tf.Variable(tf.random_normal([32]))
}
conv_W_3b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,256,128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,256,128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,128,192])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,256,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,96])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,256,64]))

}
conv_B_3b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([192])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([96])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}
conv_W_4a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,480,192])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,480,96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,96,208])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,480,16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,16,48])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,480,64]))
}
conv_B_4a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([192])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([208])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([48])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}
conv_W_4b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,160])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,112])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,112,224])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,24,64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))

}
conv_B_4b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([160])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([112])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([224])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}
conv_W_4c = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,128,256])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,24,64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))

}
conv_B_4c = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([256])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}
conv_W_4d = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,112])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,144])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,144,288])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))

}
conv_B_4d = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([112])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([144])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([288])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}
conv_W_4e = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,528,256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,528,160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,160,320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,528,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,528,128]))

}
conv_B_4e = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([128]))
}
conv_W_5a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,832,256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,832,160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,160,320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,832,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,832,128]))

}
conv_B_5a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([128]))
}
conv_W_5b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,832,384])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,832,192])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,192,384])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,832,48])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,48,128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,832,128]))

}
conv_B_5b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([384])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([192])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([384])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([48])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([128]))
}


# 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)



def do_vaild():
    out = GoogleLeNet_topological_structure(x)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    with tf.Session() as sess:
    
        saver.restore(sess, './model/train_dog_e4_1/train_dog_e4_1.model')

        summary_writer = tf.summary.FileWriter('./vaild/dog_120_e4_1', graph=tf.get_default_graph())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        fw=open('./acc/vaild_e4_1.txt','a')
        for n in range(8):
            sum_acc=0
            for i in range(num_batch):
                val, l= sess.run([img_batch, label_batch])
                summary = sess.run(merged_summary_op,feed_dict={x:val,y_:l})
                summary_writer.add_summary(summary, n*num_batch+i)
                acc = accuracy.eval({x:val, y_:l})
                sum_acc+=acc
                print(i+1, acc)
                s1 = "\r[%s%s]%.0f%%"%(">"*((n*100+i)//50)," "*(16-(n*100+i)//50),(n*100+i)/8)
                sys.stdout.write(s1)
                sys.stdout.flush()
                if((n*100+i)==799) :
                    s1 = "\r[%s%s]%s"%(">"*((n*100+i)//50)," "*(16-(n*100+i)//50),'completed')
                    print(s1)

            print(n+1,'th epoch', sum_acc/100,file=fw)
            
        fw.close()
        coord.request_stop()
        coord.join(threads)

img_batch,label_batch = load_data()
do_vaild()
            

