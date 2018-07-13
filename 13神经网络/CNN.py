# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:48 2016
@author: root
"""
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
x = tf.placeholder(tf.float32, [None, 784])                       
y_actual = tf.placeholder(tf.float32, shape=[None, 10])           
# 定义实际x与y的值。    
# placeholder中shape是参数的形状，默认为none，即一维数据，[2,3]表示为两行三列；[none，3]表示3列，行不定。

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 截尾正态分布，保留[mean-2*stddev, mean+2*stddev]范围内的随机数。用于初始化所有的权值，用做卷积核。

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 创建常量0.1；用于初始化所有的偏置项，即b，用作偏置。  

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 定义一个函数，用于构建卷积层；
# x为input；w为卷积核；strides是卷积时图像每一维的步长；padding为不同的卷积方式；

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 定义一个函数，用于构建池化层，池化层是为了获取特征比较明显的值，一般会取最大值max，有时也会取平均值mean。
# ksize=[1,2,2,1]：shape为[batch，height， width， channels]设为1个池化，池化矩阵的大小为2*2,有1个通道。
# strides是表示步长[1,2,2,1]:水平步长为2，垂直步长为2，strides[0]与strides[3]皆为1。

x_image = tf.reshape(x, [-1,28,28,1])
# 在reshape方法中-1维度表示为自动计算此维度，将x按照28*28进行图片转换，转换成一个大包下一个小包中28行28列的四维数组； 
       
W_conv1 = weight_variable([5, 5, 1, 32]) 
b_conv1 = bias_variable([32]) 
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
h_pool1 = max_pool(h_conv1) 
# 构建一定形状的截尾正态分布，用做第一个卷积核；  
# 构建一维的偏置量。    
# 将卷积后的结果进行relu函数运算，通过激活函数进行激活。 
# 将激活函数之后的结果进行池化，降低矩阵的维度。                                

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)   
h_pool2 = max_pool(h_conv2) 
# 构建第二个卷积核；
# 第二个卷积核的偏置； 
# 第二次进行激活函数运算；
# 第二次进行池化运算，输出一个2*2的矩阵，步长是2*2；                                  

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) 
# 构建新的卷积核，用来进行全连接层运算，通过这个卷积核，将最后一个池化层的输出数据转化为一维的向量1*1024。
# 构建1*1024的偏置；
# 对 h_pool2第二个池化层结果进行变形。
# 将矩阵相乘，并进行relu函数的激活。  

keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 
# 定义一个占位符,用来设置调整概率。
# 是防止过拟合的，使输入tensor中某些元素变为0，其他没变为零的元素变为原来的1/keep_prob大小，
# 形成防止过拟合之后的矩阵。            

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# 用softmax进行激励函数运算，得到预期结果；
# 在每次进行加和运算之后，需要用到激活函数进行转换，激活函数是用来做非线性变换的，因为sum出的线性函数自身在分类中存在有限性。  

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))  
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy) 
# 求交叉熵，用来检测运算结果的熵值大小。
# 通过训练获取到最小交叉熵的数据，训练权重参数。
  
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
# 计算模型的精确度。
               
sess=tf.InteractiveSession()                          
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:                 
        train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
        # 用括号中的参数，带入accuracy中，进行精确度计算。
        
        print('step',i,'training accuracy',train_acc)
        train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})
        # 训练参数，形成最优模型。

test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print("test accuracy",test_acc)






