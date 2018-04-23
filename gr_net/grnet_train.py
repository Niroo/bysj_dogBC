import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
import tensorflow.contrib.slim as slim
import grnet as gr

size=256
batch_size=8
num_batch=1000
img=[]
label=[]
n_classes=120


trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.int32, [None,n_classes])

def do_train():

    
    with slim.arg_scope(gr.net_arg_scope()):
        logits, end_points = gr.gr_net(x, n_classes,is_training=True)


    loss_1=tf.losses.softmax_cross_entropy(y_ , logits)
    loss_2=tf.losses.softmax_cross_entropy(y_ , end_points['AuxLogits'])
    cross_entropy = 0.95*loss_1+0.05*loss_2

    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(end_points['Predictions'], 1), tf.argmax(y_, 1)), tf.float32))
    
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    
    img,label=gr.read_and_decode("../tfrecord/train.tfrecords")


    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size, capacity=1000,
                                                min_after_dequeue=500)
    label_batch= slim.layers.one_hot_encoding(label_batch,n_classes)

    with tf.Session() as sess:

        
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./tmp/GN_0', graph=tf.get_default_graph())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for n in range(4):
            
            for i in range(num_batch):
                val, l= sess.run([img_batch, label_batch])

                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:val,y_:l})

                summary_writer.add_summary(summary, n*num_batch+i)
                print(i+1,"loss:",loss)



        saver.save(sess, './model/train_dog_v3rn_0/train_dog_v3rn_0.model')
        coord.request_stop()
        coord.join(threads)
        sys.exit(0)

do_train()
        
