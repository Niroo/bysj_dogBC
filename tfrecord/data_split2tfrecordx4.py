import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys


dir_path = '../test1'
dict_breed = {}
size = 256

writer = tf.python_io.TFRecordWriter("trainx4.tfrecords")
writer_x = tf.python_io.TFRecordWriter("vaildx4.tfrecords")
Index=0
for filename in os.listdir(dir_path):
    dict_breed.update({Index:filename})
    Index+=1
#print(dict_breed)


def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  


def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  

def _2tfrecord(path,label,writer):
    img = cv2.imread(path)
    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": _int64_feature(label),
        'img_raw':_bytes_feature(img_raw)
    }))
    writer.write(example.SerializeToString())

def readData(path , h = size , w = size):

    num = 0
    for i in range(1,101):
        for j in range(120):
            #print(path+'/'+dict_breed[j],i)
            img_path=path+'/'+dict_breed[j]+'/'
            _2tfrecord(img_path+str(i)+'.jpg',j,writer)
            _2tfrecord(img_path+str(i)+'_hf.jpg',j,writer)
            _2tfrecord(img_path+str(i)+'_r1.jpg',j,writer)
            _2tfrecord(img_path+str(i)+'_r2.jpg',j,writer)
        num += 1
        s1 = "\r[%s%s]%.0f%%"%(">"*(num//10)," "*(12-num//10),num/125)
        sys.stdout.write(s1)
        sys.stdout.flush()
    for n in range(101,126):
        for k in range(120):
            img_path=path+'/'+dict_breed[k]+'/'
            _2tfrecord(img_path+str(n)+'.jpg',i,writer_x)
            _2tfrecord(img_path+str(n)+'_hf.jpg',i,writer_x)
            _2tfrecord(img_path+str(n)+'_r1.jpg',i,writer_x)
            _2tfrecord(img_path+str(n)+'_r2.jpg',i,writer_x)
        num += 1
        s1 = "\r[%s%s]%.0f%%"%(">"*(num//10)," "*(12-num//10),num/1.25)
        sys.stdout.write(s1)
        sys.stdout.flush()
    if(num==125) :
        s1 = "\r[%s%s]%s"%(">"*(num//10)," "*(12-num//10),'completed')
        print(s1)
    writer.close()
    writer_x.close()


readData(dir_path)
