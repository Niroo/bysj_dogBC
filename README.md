# 基于深度卷积神经网络的犬科动物图象识别

![dataset](https://github.com/Niroo/bysj_dogBC/120dog.jpg)
## 文件结构

- xml
    - flip_data.py
    - get\_detected\_dog.py
- tfrecord
    - data_split2tfrecord.py
    - data_split2tfrecordx4.py
- gr_net
    - grnet.py
    - grnet_train.py
    - grnet_retrain.py
    - grnet_vaild.py

**get\_detected\_dog.py**

从annotation文件夹中取出bounding box对图像进行裁剪

**flip_data.py**

对图像进行数据扩充

**data_split2tfrecord.py**

将数据储存成tfrecord文件（只有原图和镜像）

**data_split2tfrecordx4.py**

将数据储存成tfrecord文件（原图、镜像、旋转、裁剪）

**grnet.py**

GRnet网络模型

**grnet_train.py**

训练网络代码

**grnet_retrain.py**

网络重训练代码

**grnet_vaild.py**

用测试集验证网络性能代码






