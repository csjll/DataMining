# -*- coding: utf-8 -*-

# 1）对数似然函数：



# 1)改进的迭代尺度法（IIS）：通过迭代的方式将对数似然函数的差值确定下界，找到其收敛性；
# 改进的迭代尺度法是将最大熵模型的对数似然形式进行优化，找到对数似然公式的下界，然后通过不停地迭代来确定最优下界的过程；
# a.对于给定的经验分布P(x,y),此时为已知量，模型参数从w——》w+sigema；对数似然函数的该变量可以确定；
# b.通过不等式变换，获取到对数似然函数的下界，然后对下界方程求偏导，求出sigema的最优解；
# c.如果w没有收敛，重复第二步；
# 注：
# 通过确定sigema函数来确定w的值，如果f(x,y)是常数，则可以直接求出sigema的值；
# 如果f(x,y)不是常数，则可以通过牛顿法迭代求出sigema的值，使g(sigema)=0,迭代sigema确定出sigema的值；
import collections
import math

class MaxEntropy():
    def __init__(self):
        self._samples = []   #样本集，元素是[y,x1,x2,...]的样本
        self._Y = set([])  #标签集合，相当去去重后的y
        self._numXY = collections.defaultdict(int)   #key为(x,y)，value为出现次数
        self._N = 0  #样本数
        self._ep_ = []   #样本分布的特征期望值
        self._xyID = {}   #key记录(x,y),value记录id号
        self._n = 0  #特征的个数
        self._C = 0   #最大特征数
        self._IDxy = {}    #key为(x,y)，value为对应的id号  
        self._w = []
        self._EPS = 0.005   #收敛条件
        self._lastw = []    #上一次w参数值
    def loadData(self,filename):
        with open(filename) as fp:
            self._samples = [item.strip().split('\t') for item in fp.readlines()]
        for items in self._samples:
                y = items[0]
                X = items[1:]
                self._Y.add(y)
                for x in X:
                    self._numXY[(x,y)] += 1
    def _sample_ep(self):   #计算特征函数fi关于经验分布的期望
        self._ep_ = [0] * self._n
        for i,xy in enumerate(self._numXY):
            self._ep_[i] = self._numXY[xy]/self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

            
        
    def _initparams(self):  #初始化参数
        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max([len(sample)-1 for sample in self._samples])
        self._w = [0]*self._n
        self._lastw = self._w[:]
        
        self._sample_ep()                 #计算每个特征关于经验分布的期望
    def _Zx(self,X):    #计算每个x的Z值
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x,y) in self._numXY:
                    ss += self._w[self._xyID[(x,y)]]
            zx += math.exp(ss)
        return zx
    
    def _model_pyx(self,y,X):   #计算每个P(y|x)
        Z = self._Zx(X)
        ss = 0
        for x in X:
            if (x,y) in self._numXY:
                ss += self._w[self._xyID[(x,y)]]
        pyx = math.exp(ss)/Z
        return pyx
       
    def _model_ep(self,index):   #计算特征函数fi关于模型的期望
        x,y = self._IDxy[index]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._model_pyx(y,sample)
            ep += pyx/self._N
        return ep
            
    def _convergence(self):
        for last,now in zip(self._lastw,self._w):
            if abs(last - now) >=self._EPS:
                return False
        return True
    
    def predict(self,X):   #计算预测概率
        Z = self._Zx(X)
        result = {}
        for y in self._Y:
            ss = 0
            for x in X:
                if (x,y) in self._numXY:
                    ss += self._w[self._xyID[(x,y)]]
            pyx = math.exp(ss)/Z
            result[y] = pyx
        return result
         
    def train(self,maxiter = 1000):   #训练数据
        self._initparams()
        for loop in range(0,maxiter):  #最大训练次数
            print ("iter:%d"%loop)
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self._model_ep(i)    #计算第i个特征的模型期望
                self._w[i] += math.log(self._ep_[i]/ep)/self._C   #更新参数
            print("w:",self._w)
            if self._convergence():   #判断是否收敛
                break
maxent = MaxEntropy()
x = ['sunny','hot','high','FALSE']
maxent.loadData('dataset.txt')
maxent.train()
print('predict::::::::::::::::::',maxent.predict(x))





# 2）拟牛顿法（BFGS）:用梯度的方式求出含有最大熵模型的对数似然函数的极大值的过程，以此来确定w的值；
# a.选定初始点w，取B0为正定对称矩阵，置k=0；
# b.计算gk=g(w(k)),若||gk||<sigema,则停止计算，确定w的值，否则进行下一步
# c.由B(k)*P(k)=-g(k),求出p(K),并计算lambda的值；
# d.w=w+lambda*p
# e.计算gk=g(w(k)),若||gk||<sigema,则停止计算，确定w的值，否则进行下一步
# f.根据B(k)计算B(K+1)的值，然后设置k=k+1,再运行c步；
import numpy as np
def bfgs(fun,gfun,hess,x0):
    #功能：用BFGS族算法求解无约束问题：min fun(x) 优化的问题请参考文章开头给出的链接
    #输入：x0是初始点，fun,gfun分别是目标函数和梯度，hess为Hessian矩阵
    #输出：x,val分别是近似最优点和最优解,k是迭代次数  
    maxk = 1e5
    rho = 0.55
    sigma = 0.4
    gama = 0.7
    epsilon = 1e-5
    k = 0
    n = np.shape(x0)[0]
    #海森矩阵可以初始化为单位矩阵
    Bk = np.eye(n) #np.linalg.inv(hess(x0)) #或者单位矩阵np.eye(n)

    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0*np.linalg.solve(Bk,gk)
        m = 0
        mk = 0
        while m < 20: # 用Wolfe条件搜索求步长
            gk1 = gfun(x0 + rho**m*dk)
            if fun(x0+rho**m*dk) < fun(x0)+sigma*rho**m*np.dot(gk,dk) and np.dot(gk1.T, dk) >=  gama*np.dot(gk.T,dk):
                mk = m
                break
            m += 1

        #BFGS校正
        x = x0 + rho**mk*dk
        print("第"+str(k)+"次的迭代结果为："+str(x))
        sk = x - x0
        yk = gfun(x) - gk   

        if np.dot(sk,yk) > 0:    
            Bs = np.dot(Bk,sk)
            ys = np.dot(yk,sk)
            sBs = np.dot(np.dot(sk,Bk),sk) 

            Bk = Bk - 1.0*Bs.reshape((n,1))*Bs/sBs + 1.0*yk.reshape((n,1))*yk/ys

        k += 1
        x0 = x

    return x0,fun(x0),k#分别是最优点坐标，最优值，迭代次数 
# x0 ,fun0 ,k = bfgs(fun,gfun,hess,np.array([3,3]))
# print(x0,fun0,k)


# 3）牛顿法：
# 是对梯度下降法的延伸，下降的梯度中包含f(x)(一次导)/f(x)(二次导)；

#文件开头加上、上面的注释。不然中文注释报错  

from numpy import *  
from math import *  
import operator  
import matplotlib  
import matplotlib.pyplot as plt  
#logistic regression ＋ 牛顿方法  
def file2matrix(filename1,filename2):#完成了文件读取、迭代运算及绘图  
    fr1 = open(filename1)#打开一个文件  
    arrayOflines1 = fr1.readlines()#返回一个行数组  
    numberOfLines1 = len(arrayOflines1)#计算行数  
    matrix = zeros((numberOfLines1,3))#生成一个全零二维数组，numberOflines1行 3列。其实数据只有两列，一列全是1.  
    row = 0  
    for line in arrayOflines1:  
        line = line.strip()#声明：s为字符串，rm为要删除的字符序列 s.strip(rm)删除s字符串中开头、结尾处，位于 rm删除序列的字符  
                            #s.lstrip(rm)       删除s字符串中开头处，位于 rm删除序列的字符  
                            #s.rstrip(rm)      删除s字符串中结尾处，位于 rm删除序列的字符  
                            #注意：#1. 当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')  
        listFromLine = line.split('  ')#将一行按（）中的参数符号分开放入一个list中  
        listFromLine[0:0] = ['1']#在list中最前面插入1，上面说了有一列全为一  
        for index,item in enumerate(listFromLine):#将list中的字符串形式的，全转换为对应的数值型。  
            listFromLine[index] = eval(item)  
        matrix[row,:] = listFromLine[:]#每个list赋给对应的二维数组的对应行  
        row+=1  
    matrix = mat(matrix)#将数组转换为矩阵  
    fr1.close()  
    fr2 = open(filename2)  
    arrayOflines2 = fr2.readlines()  
    numberOfLines2 = len(arrayOflines2)  
    matrixy = zeros((numberOfLines2,1))  
    row = 0;  
    for line in arrayOflines2:  
        line = line.strip()  
        listFromLine = [line]  
        for index,item in enumerate(listFromLine):  
            listFromLine[index] = eval(item)  
        matrixy[row,:] = listFromLine[:]  
        row+=1  
    matrixy = mat(matrixy)  
    fr2.close()  
    tempxxt = dot(matrix.T,matrix).I#这一部分乘上下面的denominator()既为H.I（Hessian矩阵的逆）  
    theta=mat(zeros((3,1)))#初始θ参数，全零  
    for i in range(0,2000):#迭代2000次得到了比较好的结果。我采取的可能是全向量的形式计算，感觉迭代次数有点偏多  
        temphypo=Hypothesis(theta,matrix,row)  
        tempdenominator=denominator(temphypo,row)  
        tempnumerator=numerator(temphypo,matrixy,matrix)  
        theta = theta+dot(tempxxt,tempnumerator)/tempdenominator  
    temparray = ravel(Hypothesis(theta,matrix,row))  
    temptheta = ravel(theta)  
    for i in range(0,row):#根据hypothesis函数的值进行标记,方便绘图  
        if(temparray[i]>=0.5):  
            temparray[i]=1  
        else:  
            temparray[i]=0;  
    fig = plt.figure()#生成了一个图像窗口  
    ax = fig.add_subplot(111)#刚才窗口中的一个子图  
    ax.scatter(ravel(matrix[:,1]),ravel(matrix[:,2]),200,20*temparray)#生成离散点，参数分别为点的x坐标数组、y坐标数组、点的大小数组、点的颜色数组  
    x=linspace(-1,10,100)#起点为－1，终止10，100个元素的等差数组x  
    ax.plot(x,-(temptheta[0]+temptheta[1]*x)/temptheta[2])#绘制x为自变量的函数曲线  
    plt.show()  
  
def Hypothesis(theta,x,row):#假设函数、是一个向量形式  
    hypo = zeros((row,1))  
    for i in range(0,row):  
        temp = exp(-dot(theta.T,x[i].T))  
        hypo[i,:] = [1/(1+temp)]  
    return hypo  
  
def denominator(hypo,row):#分母部分  
    temp=zeros((row,1))  
    temp.fill(1)  
    temp=temp-hypo  
    temp=dot(hypo.T,temp)  
    return temp  
  
def numerator(hypo,y,x):#牛顿方法的分子，我们要做的就是迭代使这一部分接近零  
    temp = y-hypo  
    temp = dot(temp.T,x)  
    return temp.T  


# 4）拉格朗日乘数法





















