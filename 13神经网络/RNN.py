# -*- coding: utf-8 -*-

import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)     

lr = 0.001
training_iters = 100000
batch_size = 128
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

# 生成两个占位符；
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # 随机生成一个符合正态图形的矩阵，作为in和out的初始值。
    'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out':tf.Variable(tf.random_normal(n_hidden_units, n_classes)),
    }

biases = {
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes, ])),
    }

def RNN(X, weights, biases):
    # 第一步：输入的x为三维数据，因此需要进行相应的维度变换；转换成2维，然后与w、b进行交易，运算完成后，再将x转换成三维；
    X=tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in'])+biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    # 第二步：即构建cell的初始值，并进行建模运算；
    # n_hidden_units:是ht的维数，表示128维行向量；state_is_tuple表示tuple形式，返回一个lstm的单元，即一个ht。
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 将LSTM的状态初始化全为0数组，batch_size给出一个batch大小。
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # 运算一个神经单元的输出值与状态，动态构建RNN模型，在这个模型中实现ht与x的结合。
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    
    # 第三步：将输出值进行格式转换，然后运算输出，即可。
    # 矩阵的转置，[0,1,2]为正常顺序[高，长，列]，想要更换哪个就更换哪个的顺序即可,并实现矩阵解析。
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    return results
# 创建一个模型，然后进行测试。 
pred = RNN(x, weights, biases)
# softmax_cross_entropy_with_logits：将神经网络最后一层的输出值pred与实际标签y作比较，然后计算全局平均值，即为损失。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# 用梯度下降优化，下降速率为0.001。
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
# 计算准确度。
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x:batch_xs, 
            y:batch_ys,
            })
        
        if step % 20 ==0:
            print(sess.run(accuracy, feed_dict={
                x:batch_xs,
                y:batch_ys,
                }))
            
            step += 1




# RNN案例（二）
# num_epochs = 100    
# total_series_length = 50000    
# truncated_backprop_length = 15    
# state_size = 4    
# num_classes = 2    
# echo_step = 3    
# batch_size = 5    
# num_batches = total_series_length//batch_size//truncated_backprop_length    
# 
# def generateData():    
#     x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))    
#     y = np.roll(x, echo_step)    
#     y[0:echo_step] = 0    
#     
#     x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows    
#     y = y.reshape((batch_size, -1))    
#     
#     return (x, y)    
# 
# batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])    
# batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])    
# 
# init_state = tf.placeholder(tf.float32, [batch_size, state_size])    
# 
# W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)    
# b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)    
# 
# W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)    
# b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)    
# 
# # Unpack columns    
# inputs_series = tf.unstack(batchX_placeholder, axis=1)    
# labels_series = tf.unstack(batchY_placeholder, axis=1)    
# 
# # Forward pass    
# current_state = init_state    
# states_series = []    
# for current_input in inputs_series:    
#     current_input = tf.reshape(current_input, [batch_size, 1])    
#     input_and_state_concatenated = tf.concat(1, [current_input, current_state])  # Increasing number of columns    
# 
#     next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition    
#     states_series.append(next_state)    
#     current_state = next_state    
# 
# logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition    
# predictions_series = [tf.nn.softmax(logits) for logits in logits_series]    
# 
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]    
# total_loss = tf.reduce_mean(losses)    
# 
# train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)    
# 
# def plot(loss_list, predictions_series, batchX, batchY):    
#     plt.subplot(2, 3, 1)    
#     plt.cla()    
#     plt.plot(loss_list)    
# 
#     for batch_series_idx in range(5):    
#         one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]    
#         single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])    
#         
#         plt.subplot(2, 3, batch_series_idx + 2)    
#         plt.cla()    
#         plt.axis([0, truncated_backprop_length, 0, 2])    
#         left_offset = range(truncated_backprop_length)    
#         plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")    
#         plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")    
#         plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")    
# 
#     plt.draw()    
#     plt.pause(0.0001)    
# 
# with tf.Session() as sess:    
#     sess.run(tf.initialize_all_variables())    
#     plt.ion()    
#     plt.figure()    
#     plt.show()    
#     loss_list = []    
# 
# for epoch_idx in range(num_epochs):    
#     x,y = generateData()    
#     _current_state = np.zeros((batch_size, state_size))    
#     print("New data, epoch", epoch_idx)    
# 
# for batch_idx in range(num_batches):    
#     start_idx = batch_idx * truncated_backprop_length    
#     end_idx = start_idx + truncated_backprop_length    
#     
#     batchX = x[:,start_idx:end_idx]    
#     batchY = y[:,start_idx:end_idx]    
# 
#     _total_loss, _train_step, _current_state, _predictions_series = sess.run(    
#     [total_loss, train_step, current_state, predictions_series],    
#     feed_dict={    
#     batchX_placeholder:batchX,    
#     batchY_placeholder:batchY,    
#     init_state:_current_state    
#     })    
# 
#     loss_list.append(_total_loss)    
# 
#     if batch_idx%100 == 0:    
#         print("Step",batch_idx, "Loss", _total_loss)    
#         plot(loss_list, _predictions_series, batchX, batchY)    
# 
#     plt.ioff()    
#     plt.show()
