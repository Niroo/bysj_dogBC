# -*- codeing: utf-8 -*-
import sys
import os
import cv2
import xml.etree.ElementTree as ET

input_dir = '../Images'
anno_dir= '../Annotation'
output_dir = '../de_Img'
size = 256


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


num=0
for (path, dirnames, filenames) in os.walk(input_dir):
        if not(path == input_dir):
            if not os.path.exists(output_dir+'/'+path.split('-',1)[-1]):
                os.makedirs(output_dir+'/'+path.split('-',1)[-1])
            anno=anno_dir+'/'+path.split('/')[-1]
            index = 1
            num += 1
            #print(path)
            for filename in filenames:
                if filename.endswith('.jpg'):
                    #print('Being processed picture {0}  '.format(index)+path.split('-',1)[-1] )
                    img_path = path+'/'+filename
                    
                    # 获取 XML 文档对象 ElementTree
                    tree = ET.parse(anno+'/'+filename.split('.')[0])
                    # 获取 XML 文档对象的根结点 Element
                    root = tree.getroot()

                    img = cv2.imread(img_path)
                    # 转为灰度图片
                    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    xmin=int(root[5][4][0].text)
                    ymin=int(root[5][4][1].text)
                    xmax=int(root[5][4][2].text)
                    ymax=int(root[5][4][3].text)

                    do_img = img[ymin:ymax,xmin:xmax]
                    
                    do_img=cv2.resize(do_img,(size,size))
                    # 保存图片
                    cv2.imwrite(output_dir+'/'+path.split('-',1)[-1]+'/'+str(index)+'.jpg', do_img)
                    index += 1
                    #cv2.imshow('image',do_img)
                    #key = cv2.waitKey(30) & 0xff
                    #if key == 27:
                    #    sys.exit(0)
        s1 = "\r[%s%s]%.0f%%"%(">"*(num//10)," "*(12-num//10),num/1.2)
        sys.stdout.write(s1)
        sys.stdout.flush()
        if(num==120) :
            s1 = "\r[%s%s]%s"%(">"*(num//10)," "*(12-num//10),'completed')
            print(s1)

