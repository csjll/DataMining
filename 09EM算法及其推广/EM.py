# -*- coding: utf-8 -*-
# EM算法：E：猜(求期望)；M：反思(求极值)；
# 1）获取一些基本的假设，一般采用最大熵假设，均匀分布概率；M步（假设步）；假设有问题的概率；
# 2）即根据一些假设，先猜其发生概率，此步猜测是基于全局的；E步（观察步）；计算有问题的次数；
#   进行一定程度的观察，去除一些没有发生的数据，计算发生的次数，此步是基于观察的，用于缩小范围；
# 3）然后根据观察出来的数据重新猜测，即重新计算概率，此步是基于上步中缩小范围的；M步；在计算的次数的基础上计算概率；
# 4）在更小的范围中重新计算次数，此步是基于缩小范围的；E步（观察步）；再在计算的概率的基础上计算次数；
# 5）然后再在新次数的基础上计算概率，以进一步缩小范围；M步；再在计算的次数的基础上计算概率；
#   然后循环，最后你得到了一个可以解释整个数据的假设；

#! /usr/bin/env python  
#coding=utf-8  
''''' 
author:zhaojiong 
EM算法初稿2016-4-28 
初始化三个一维的高斯分布， 
 
'''  
from numpy import *  
import numpy as np  
import matplotlib.pyplot as plt  
import copy   
  
def init_em(x_num=2000):  
    '''
             初始化数据
    '''  
    global  mod_num,mod_prob_arr,x_prob_mat,theta_mat,theta_mat_temp,x_mat,mod_prob_arr_test  
    mod_num=3  
    x_mat =zeros((x_num,1))  
    mod_prob_arr=[0.3,0.4,0.3] #三个状态  
    mod_prob_arr_test=[0.3,0.3,0.4]  
      
    x_prob_mat=zeros((x_num,mod_num))  
    #theta_mat =zeros((mod_num,2))  
    theta_mat =array([ [30.0,4.0],  
                       [80.0,9.0],  
                       [180.0,3.0]  
                    ])  
    theta_mat_temp =array([ [20.0,3.0],  
                            [60.0,7.0],  
                            [80.0,2.0]  
                            ])  
    for i in range(x_num):  
        if np.random.random(1)<=mod_prob_arr[0]:  
            x_mat[i,0] = np.random.normal()*math.sqrt(theta_mat[0,1]) + theta_mat[0,0]  
        elif np.random.random(1)<= mod_prob_arr[0]+mod_prob_arr[1]:  
            x_mat[i,0] = np.random.normal()*math.sqrt(theta_mat[1,1]) + theta_mat[1,0]  
        else:   
            x_mat[i,0] = np.random.normal()*math.sqrt(theta_mat[2,1]) + theta_mat[2,0]  
      
    return x_mat 
 
def plot_data(x_mat):  
    plt.hist(x_mat[:,0],200)  
    plt.show()  
      
def e_step(x_arr):  
    x_row, x_colum =shape(x_arr)  
    global  mod_num,mod_prob_arr,x_prob_mat,theta_mat,theta_mat_temp,mod_prob_arr_test  
    for i in range(x_row):  
        Denom = 0.0  
        for j in range(mod_num):  
            exp_temp=math.exp((-1.0/(2*(float(theta_mat_temp[j,1]))))*(float(x_arr[i,0]-theta_mat_temp[j,0]))**2)  
              
            Denom += mod_prob_arr_test[j]*(1.0/math.sqrt(theta_mat_temp[j,1]))*exp_temp  
          
        for j in range(mod_num):  
            Numer = mod_prob_arr_test[j]*(1.0/math.sqrt(theta_mat_temp[j,1]))*math.exp((-1.0/(2*(float(theta_mat_temp[j,1]))))*(float(x_arr[i,0]-theta_mat_temp[j,0]))**2)  
#            if(Numer<1e-6):  
#                Numer=0.0  
            if(Denom!=0):  
                x_prob_mat[i,j] = Numer/Denom  
            else:  
                x_prob_mat[i,j]=0.0  
    return x_prob_mat  

def m_step(x_arr):  
    x_row, x_colum =shape(x_arr)  
    global  mod_num,mod_prob_arr,x_prob_mat,theta_mat,theta_mat_temp,mod_prob_arr_test  
    for j in range(mod_num):  
        MU_K = 0.0  
        Denom = 0.0  
        MD_K=0.0  
        for i in range(x_row):  
            MU_K += x_prob_mat[i,j]*x_arr[i,0]  
            Denom +=x_prob_mat[i,j]   
             
        theta_mat_temp[j,0] = MU_K / Denom   
        for i in range(x_row):  
            MD_K +=x_prob_mat[i,j]*((x_arr[i,0]-theta_mat_temp[j,0])**2)  
          
        theta_mat_temp[j,1] = MD_K / Denom  
        mod_prob_arr_test[j]=Denom/x_row  
    return theta_mat_temp 
 
def main_run(iter_num=500,Epsilon=0.0001,data_num=2000):  
    init_em(data_num)  
    global  mod_num,mod_prob_arr,x_prob_mat,theta_mat,theta_mat_temp,x_mat,mod_prob_arr_test  
    theta_row, theta_colum =shape(theta_mat_temp)  
    for i in range(iter_num):  
        Old_theta_mat_temp=copy.deepcopy(theta_mat_temp)  
        x_prob_mat=e_step(x_mat)  
        theta_mat_temp= m_step(x_mat)  
        if sum(abs(theta_mat_temp-Old_theta_mat_temp)) < Epsilon:  
           print("第 %d 次迭代退出" %i)
           break
    return theta_mat_temp  
  
def test(data_num):  
    testdata=init_em(data_num)  
    #print(testdata)   
    #print('\n')  
    plot_data(testdata)  

# 高斯混合模型：
# 1）取参数的初始值开始迭代；
# 2）E步：依据当前模型参数，计算分模型k对观测数据y的响应度，r为每个高斯模型对观测值的贡献程度；
# 3）M步：计算新一轮迭代的模型参数u、xigema、a，三个参数为各个独立高斯的参数；

# 此示例程序随机从4个高斯模型中生成500个2维数据，
# 真实参数：
# 混合项w=[0.1，0.2，0.3，0.4]，
# 均值u=[[5，35]，[30，40]，[20，20]，[45，15]]，
# 协方差矩阵∑=[[30，0]，[0，30]]。
# 然后以这些数据作为观测数据，
# 根据EM算法来估计以上参数（此程序未估计协方差矩阵）


import math  
import copy  
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
  
#生成随机数据，4个高斯模型  
def generate_data(sigma,N,mu1,mu2,mu3,mu4,alpha):  
    global X                  #可观测数据集  
    X = np.zeros((N, 2))       # 初始化X，2行N列。2维数据，N个样本  
    X=np.matrix(X)  
    global mu                 #随机初始化mu1，mu2，mu3，mu4  
    mu = np.random.random((4,2))  
    mu=np.matrix(mu)  
    global excep              #期望第i个样本属于第j个模型的概率的期望  
    excep=np.zeros((N,4))  
    global alpha_             #初始化混合项系数  
    alpha_=[0.25,0.25,0.25,0.25]  
    for i in range(N):  
        if np.random.random(1) < 0.1:  # 生成0-1之间随机数  
            X[i,:]  = np.random.multivariate_normal(mu1, sigma, 1)     #用第一个高斯模型生成2维数据  
        elif 0.1 <= np.random.random(1) < 0.3:  
            X[i,:] = np.random.multivariate_normal(mu2, sigma, 1)      #用第二个高斯模型生成2维数据  
        elif 0.3 <= np.random.random(1) < 0.6:  
            X[i,:] = np.random.multivariate_normal(mu3, sigma, 1)      #用第三个高斯模型生成2维数据  
        else:  
            X[i,:] = np.random.multivariate_normal(mu4, sigma, 1)      #用第四个高斯模型生成2维数据  
  
    print("可观测数据：\n",X)       #输出可观测样本  
    print("初始化的mu1，mu2，mu3，mu4：",mu)      #输出初始化的mu  
  
def e_step(sigma,k,N):  
    global X  
    global mu  
    global excep  
    global alpha_  
    for i in range(N):  
        denom=0  
        for j in range(0,k):  
            denom += alpha_[j]*math.exp(-(X[i,:]-mu[j,:])*sigma.I*np.transpose(X[i,:]-mu[j,:]))/np.sqrt(np.linalg.det(sigma))       #分母  
        for j in range(0,k):  
            numer = math.exp(-(X[i,:]-mu[j,:])*sigma.I*np.transpose(X[i,:]-mu[j,:]))/np.sqrt(np.linalg.det(sigma))        #分子  
            excep[i,j]=alpha_[j]*numer/denom      #求期望  
    print("隐藏变量：\n",excep)  
  
def m_step(k,N):  
    global excep  
    global X  
    global alpha_  
    for j in range(0,k):  
        denom=0   #分母  
        numer=0   #分子  
        for i in range(N):  
            numer += excep[i,j]*X[i,:]  
            denom += excep[i,j]  
        mu[j,:] = numer/denom    #求均值  
        alpha_[j]=denom/N        #求混合项系数  
  
if __name__ == '__main__':  
    iter_num=1000  #迭代次数  
    N=500         #样本数目  
    k=4            #高斯模型数  
    probility = np.zeros(N)    #混合高斯分布  
    u1=[5,35]  
    u2=[30,40]  
    u3=[20,20]  
    u4=[45,15]  
    sigma=np.matrix([[30, 0], [0, 30]])               #协方差矩阵  
    alpha=[0.1,0.2,0.3,0.4]         #混合项系数  
    generate_data(sigma,N,u1,u2,u3,u4,alpha)     #生成数据  
    #迭代计算  
    for i in range(iter_num):  
        err=0     #均值误差  
        err_alpha=0    #混合项系数误差  
        Old_mu = copy.deepcopy(mu)  
        Old_alpha = copy.deepcopy(alpha_)  
        e_step(sigma,k,N)     # E步  
        m_step(k,N)           # M步  
        print("迭代次数:",i+1)  
        print("估计的均值:",mu)  
        print("估计的混合项系数:",alpha_)  
        for z in range(k):  
            err += (abs(Old_mu[z,0]-mu[z,0])+abs(Old_mu[z,1]-mu[z,1]))      #计算误差  
            err_alpha += abs(Old_alpha[z]-alpha_[z])  
        if (err<=0.001) and (err_alpha<0.001):     #达到精度退出迭代  
            print(err,err_alpha)  
            break  
    #可视化结果  
    # 画生成的原始数据  
    plt.subplot(221)  
    plt.scatter(X[:,0], X[:,1],c='b',s=25,alpha=0.4,marker='o')    #T散点颜色，s散点大小，alpha透明度，marker散点形状  
    plt.title('random generated data')  
    #画分类好的数据  
    plt.subplot(222)  
    plt.title('classified data through EM')  
    order=np.zeros(N)  
    color=['b','r','k','y']  
    for i in range(N):  
        for j in range(k):  
            if excep[i,j]==max(excep[i,:]):  
                order[i]=j     #选出X[i,:]属于第几个高斯模型  
            probility[i] += alpha_[int(order[i])]*math.exp(-(X[i,:]-mu[j,:])*sigma.I*np.transpose(X[i,:]-mu[j,:]))/(np.sqrt(np.linalg.det(sigma))*2*np.pi)    #计算混合高斯分布  
        plt.scatter(X[i, 0], X[i, 1], c=color[int(order[i])], s=25, alpha=0.4, marker='o')      #绘制分类后的散点图  
    #绘制三维图像  
    ax = plt.subplot(223, projection='3d')  
    plt.title('3d view')  
    for i in range(N):  
        ax.scatter(X[i, 0], X[i, 1], probility[i], c=color[int(order[i])])  
    plt.show() 

