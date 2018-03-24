import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

dir_path = '../test'
dict_breed = {}
size = 256

writer = tf.python_io.TFRecordWriter("train.tfrecords")
writer_x = tf.python_io.TFRecordWriter("vaild.tfrecords")
Index=1
for filename in os.listdir(dir_path):
    dict_breed.update({filename:Index})
    Index+=1
#print(dict_breed)

def _int64_feature(value):  
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
  
def _bytes_feature(value):  
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  

def readData(path , h=size, w=size):
    index = 0
    num = 0
    for (root, dirnames, filenames) in os.walk(dir_path):
        if not(root == dir_path):
            print(root)
            num += 1
            for filename in filenames:
                if filename.endswith('_hf.jpg'):
                    index += 1
                    img_path = root+'/'+filename
                    # 从文件读取图片
                    img = cv2.imread(img_path)
                    img_o = cv2.imread(root+'/'+filename.split('_')[0]+'.jpg')
                    #cv2.imshow('image',img_o)
                    #key = cv2.waitKey(30) & 0xff
                    #if key == 27:
                    #   sys.exit(0)
                    img_raw = img.tobytes()
                    img_raw_o = img_o.tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": _int64_feature(dict_breed[root.split('/')[-1]]-1),
                        'img_raw':_bytes_feature(img_raw)
                    }))
                    
                    example_o = tf.train.Example(features=tf.train.Features(feature={
                        "label": _int64_feature(dict_breed[root.split('/')[-1]]-1),
                        'img_raw':_bytes_feature(img_raw_o)
                    }))
                    if index%5==0:
                        writer_x.write(example.SerializeToString())
                        writer_x.write(example_o.SerializeToString())
                    else:
                        writer.write(example.SerializeToString())
                        writer.write(example_o.SerializeToString())
        s1 = "\r[%s%s]%.0f%%"%(">"*num*2," "*(12-num*2),num/6)
        sys.stdout.write(s1)
        sys.stdout.flush()
        if(num==6) :
            s1 = "\r[%s%s]%s"%(">"*num*2," "*(12-num*2),'completed')
            print(s1)
    writer.close()

readData(dir_path)
