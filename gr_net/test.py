import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
import tensorflow.contrib.slim as slim

size=256
batch_size=10
num_batch=100
img=[]
label=[]
n_classes=120
total_layers = 25 #resnet's deep
units_between_stride = total_layers // 5


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
    #label = tf.one_hot(label, depth= n_classes)
    #label = tf.reshape(label, [_, n_classes])
    #print(label)

    return img, label

img,label=read_and_decode("../tfrecord/train.tfrecords")


img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)
label_batch= slim.layers.one_hot_encoding(label_batch,n_classes)


x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.int32, [None,n_classes])

def resUnit(input_layer,channel,csize,i):
    with tf.variable_scope("res_unit"+str(i)):
        part1 = slim.batch_norm(input_layer,activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2,channel,[csize,csize],activation_fn=None)
        part4 = slim.batch_norm(part3,activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5,channel,[csize,csize],activation_fn=None)
        output = input_layer + part6
        return output

def resneXt_unit(input_layer,channel,csize,i):
    with tf.variable_scope("resneXt_unit"+str(i)):
        net=[]
        for i in range(32):    
            A1 = slim.batch_norm(input_layer,activation_fn=None)
            A2 = tf.nn.relu(A1)
            A3 = slim.conv2d(A2,4,[1,1],activation_fn=None)
            A4 = slim.batch_norm(A3,activation_fn=None)
            A5 = tf.nn.relu(A4)
            A6 = slim.conv2d(A5,4,[3,3],activation_fn=None)
            net.append(A6)
        output = tf.concat(net,3)
        output = slim.conv2d(output,channel,[csize,csize],activation_fn=None)
        output = input_layer + output
        return output

weights = {
    'conv1_7x7_S2': tf.Variable(tf.random_normal([7,7,3,64])),
    'conv2_1x1_S1': tf.Variable(tf.random_normal([1,1,64,64])),
    'conv2_3x3_S1': tf.Variable(tf.random_normal([3,3,64,192])),
    'FC2': tf.Variable(tf.random_normal([8*8*1024, 120]))
}

biases = {
    'conv1_7x7_S2': tf.Variable(tf.random_normal([64])),
    'conv2_1x1_S1': tf.Variable(tf.random_normal([64])),
    'conv2_3x3_S1': tf.Variable(tf.random_normal([192])),
    'FC2': tf.Variable(tf.random_normal([120]))

}
pooling = {
    'pool1_3x3_S2': [1,3,3,1],
    'pool2_3x3_S2': [1,3,3,1],
    'pool3_3x3_S2': [1,3,3,1],
    'pool4_3x3_S2': [1,3,3,1],
}

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
    
    for i in range(units_between_stride):
        x=resUnit(x,64,3,i)
    x = slim.conv2d(x,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_down_10')
    # [32,32,64]
    # A2
    x = resneXt_unit(x,64,3,1)
    x = slim.conv2d(x,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_down_11')
    # [16,16,64]

    x = resneXt_unit(x,64,3,2)
    x = slim.conv2d(x,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_down_12')
    # [8,8,64]
    layer1 = x
    for i in range(3):    
        layer1 = resUnit(layer1,64,3,i+100)
        layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_down_'+str(i))
    top = slim.conv2d(layer1,n_classes,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')
    output = slim.layers.softmax(slim.layers.flatten(top))
    return output


def cnnTrain():
    
    out = GoogleLeNet_topological_structure(x)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.03).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./tmp/dog_120_e4', graph=tf.get_default_graph())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for n in range(5):
             # 每次取batch_size张图片
            for i in range(num_batch):
                val, l= sess.run([img_batch, label_batch])
                # 开始训练数据，同时训练三个变量，返回三个数据
                #print(val)
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:val,y_:l})
                summary_writer.add_summary(summary, n*num_batch+i)
                print(i+1,"loss:",loss)
            if not os.path.exists('./acc'):
                os.makedirs('./acc')
            sum_acc=0
            fw=open('./acc/acc_e4.txt','a')
            for j in range(10):
                val, l= sess.run([img_batch, label_batch])   
                acc = accuracy.eval({x:val, y_:l})
                sum_acc+=acc
                
                print(j+1, acc,file=fw)
                
            print(n+1, sum_acc/10)
            print(n+1,'th epoch', sum_acc/10,file=fw)
            fw.close()
        saver.save(sess, './model/train_dog_e4.model')
        coord.request_stop()
        coord.join(threads)
        sys.exit(0)

cnnTrain()
        
