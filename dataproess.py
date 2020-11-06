import tensorflow as tf
import os
import matplotlib.pyplot as plt
import torch

from PIL import Image
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_dir = "E:\dataset"
#读文件
train_short_dir1 = data_dir + "/01/"+ "/1/"
train_normal_dir2 = data_dir + "/01/"+ "/2/"
train_height_dir3 = data_dir + "/01/"+ "/3/"
train_tfrecord_file = data_dir + "/train.tfrecords"
test_short_dir1 = data_dir + "/02/"+ "/1/"
test_normal_dir2 = data_dir + "/02/"+ "/2/"
test_height_dir3 = data_dir + "/02/"+ "/3/"
test_tfrecord_file = data_dir + "/test.tfrecords"

train_short_filenames1 = [train_short_dir1 + filename for filename in os.listdir(train_short_dir1)]
train_normal_filenames2 = [train_normal_dir2 + filename for filename in os.listdir(train_normal_dir2)]
train_height_filenames3 = [train_height_dir3 + filename for filename in os.listdir(train_height_dir3)]
test_short_filenames1 = [test_short_dir1 + filename for filename in os.listdir(test_short_dir1)]
test_normal_filenames2 = [test_normal_dir2 + filename for filename in os.listdir(test_normal_dir2)]
test_height_filenames3 = [test_height_dir3 + filename for filename in os.listdir(test_height_dir3)]

train_filenames =train_short_filenames1+train_normal_filenames2+train_height_filenames3
test_filenames =test_short_filenames1+test_normal_filenames2+test_height_filenames3
train_labels = [1] * len(train_short_filenames1)+[2] * len(train_normal_filenames2)+[3] * len(train_height_filenames3)
test_labels = [1] * len(test_short_filenames1)+[2] * len(test_normal_filenames2)+[3] * len(test_height_filenames3)

#提取特征
def  get_hrate(filenames):
        img1 = plt.imread(filenames, 0)
        longh=img1.shape[0]
        ret, binary = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            return h/longh
            break

def  get_wrate(filenames):
    img1 = plt.imread(filenames, 0)
    width1 = img1.shape[1]
    #print(img1.shape)
    ret, binary = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        return w/width1
        break
def  get_whrate(filenames):
    img1 = plt.imread(filenames, 0)
    ret, binary = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        #print(float(w/h))
        return float(w/h)
        break

def encoder(filenames, labels, tfrecord_file):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for filename, label in zip(filenames, labels):
            # 提取特征
            #heightfesture=get_hrate(filename)
            #widthfeature=get_wrate(filename)
            whrateteture=get_whrate(filename)
            feature = {  # 建立feature字典
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'whrate': tf.train.Feature(float_list=tf.train.FloatList(value=[whrateteture])),
                #'wrate': tf.train.Feature(float_list=tf.train.FloatList(value=[widthfeature])),
                #'hwrate':tf.train.Feature(float_list=tf.train.FloatList(value=[get_whrate(filename)]))
            }


            # 通过字典创建example
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # 将example序列化并写入字典
            writer.write(example.SerializeToString())


encoder(train_filenames, train_labels, train_tfrecord_file)
encoder(test_filenames, test_labels, test_tfrecord_file)
#print(train_labels)
#print(len(train_filenames))


def decoder(tfrecord_file, is_train_dataset=None):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    #print(dataset)
    feature_discription = {
        #'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'whrate': tf.io.FixedLenFeature([], tf.float32),
         #'wrate': tf.io.FixedLenFeature([], tf.float32),
        #'whrate': tf.io.FixedLenFeature([], tf.float32)
    }

    def _parse_example(example_string):  # 解码每一个example
        feature_dic = tf.io.parse_single_example(example_string, feature_discription)
        return feature_dic['whrate'],feature_dic['label']

    batch_size = 1

    if is_train_dataset is not None:
        dataset = dataset.map(_parse_example).shuffle(buffer_size=2000).batch(batch_size).prefetch(
            tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(_parse_example)
        dataset = dataset.batch(batch_size)

    return dataset
train_data = decoder(train_tfrecord_file, 1)
test_data = decoder(test_tfrecord_file)

#train_data = decoder(train_tfrecord_file, 1)
#test_data = decoder(test_tfrecord_file)
#print(train_data)

lsvm = SVC(kernel='rbf', C=1.0,gamma=0.1)
listtrainh=[]
listtrainw=[]
listtrainlabel=[]
listtesth=[]
listtestw=[]
listtestlabel=[]
#训练集、测试集大小
#处理训练集
train_data2=list(train_data.as_numpy_iterator())
#print(train_data2)
for batch, (x, y) in enumerate(train_data2):
    listtrainh.append(y)
    listtrainlabel.append(x)
    #print(listtrainlabel)
#处理测试集
test_data3=list(test_data.as_numpy_iterator())
print(' 训练集',len(train_data2),' 测试集',len(test_data3))
for batch, (x, y,) in enumerate(test_data3):
    listtesth.append(x)
    listtestlabel.append(y)
    #print(listtestlabel)
lsvm.fit(listtrainlabel,listtrainh)
print ('训练集准确率：',lsvm.score(listtrainlabel, listtrainh))
#print ('训练集准确率：', accuracy_score(listtrain, lsvm.predict(listtrainlabel)))
lsvm.predict(listtesth)
print ('测试集准确率：',lsvm.score(listtesth, listtestlabel))
#print ('测试集准确率：', accuracy_score(listtestlabel, lsvm.predict(listtest)))

