# -*- codeing: utf-8 -*-
import sys
import os
import cv2
import dlib

input_dir = './Images'
output_dir = './test'
size = 256

if not os.path.exists(output_dir):
    os.makedirs(output_dir)



for (path, dirnames, filenames) in os.walk(input_dir):
        if not(path == input_dir):
            if not os.path.exists(output_dir+'/'+path.split('-')[-1]):
                os.makedirs(output_dir+'/'+path.split('-')[-1])
            index = 1
            print(path)
            for filename in filenames:
                if filename.endswith('.jpg'):
                    print('Being processed picture {0}  '.format(index)+path.split('-')[-1] )
                    img_path = path+'/'+filename
                    # 从文件读取图片
                    img = cv2.imread(img_path)
                    # 转为灰度图片
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    gray_img = cv2.resize(gray_img, (size,size))
                    cv2.imshow('image',gray_img)
                    # 保存图片
                    cv2.imwrite(output_dir+'/'+path.split('-')[-1]+'/'+str(index)+'.jpg', gray_img)
                    index += 1

                    key = cv2.waitKey(30) & 0xff
                    if key == 27:
                        sys.exit(0)

