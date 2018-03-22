# -*- codeing: utf-8 -*-
import sys
import os
import cv2

input_dir = '../de_Img'
output_dir = '../Img_extend'
size = 256


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


num = 0
for (path, dirnames, filenames) in os.walk(input_dir):
        if not(path == input_dir):
            if not os.path.exists(output_dir+'/'+path.split('/')[-1]):
                os.makedirs(output_dir+'/'+path.split('/')[-1])
            index = 1
            num += 1
            
            #print(path)
            for filename in filenames:
                if filename.endswith('.jpg'):
                    #print('Being processed picture {0}  '.format(index)+path.split('/')[-1])
                    img_path = path+'/'+filename

                    img = cv2.imread(img_path)
                    # 转为灰度图片
                    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    do_img = cv2.flip(img,1)
                    # 保存图片
                    cv2.imwrite(output_dir+'/'+path.split('/')[-1]+'/'+str(index)+'.jpg', img)
                    cv2.imwrite(output_dir+'/'+path.split('/')[-1]+'/'+str(index)+'_hf.jpg', do_img)
                    #print(output_dir+'/'+path.split('/')[-1]+'/'+str(index)+'.jpg')
                    index += 1
        s1 = "\r[%s%s]%.0f%%"%(">"*(num//10)," "*(12-num//10),num/1.2)
        sys.stdout.write(s1)
        sys.stdout.flush()
        if(num==120) :
            s1 = "\r[%s%s]%s"%(">"*(num//10)," "*(12-num//10),'completed')
            print(s1)                  

