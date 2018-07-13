# -*- coding: utf-8 -*-

# 马尔科夫模型：
# 即一个事件的发生只依赖于前一个事件，多用于时间序列事件；
# P(Xt+1/(Xt,Xt-1,Xt-2……))=P(Xt+1/Xt)
# 可以计算并行案例的概率P(x1,x2,x3……)=P(x2/x1)*P(x3/x2)……

# 隐含马尔科夫模型：
# 即在原来马尔科夫假设的基础上增加隐含层；
# xi只取决于yi,yi取决于y(i-1);x为显性的；y为隐性的；通过观察x来推测y；
# 故需要求隐含层各个事件的发生状态：
# P(y1,y2,y3)=sum(sum(sum(P(y1,y2,y3,x1,x2,x3))))
#            =sum(sum(sum(P(y1/y2,y3,x1,x2,x3)*P(y2,y3,x1,x2,x3))))
#            ……
#            =sum(sum(sum(P(y3/x3)*P(x3/x2)*P(y2/x2)*P(x2/x1)*P(y1/x1)*P(x1))))
# 即：
# 隐含层的概率运算=xi的条件下yi的概率与x(i-1)的条件下xi的概率的乘积之和；

# 前向后向算法：
#     在已知初始概率、隐藏层的转化概率、观察层的显示概率的情况下，如何计算显示层的联合概率；
#     1)计算各个步骤的转移概率，A、B、O、I四个矩阵；
#     2)计算各个步骤的前向概率和后向概率；
#     3)将最后一步前向概率求和或者最前一步后向概率求和即可获得观测序列概率；

# 鲍姆-韦尔奇算法：
#     即通过EM算法迭代的方式，来计算HMM的三个常用参数：初始概率、隐藏层的转化概率、观察层的显示概率；
#     1）首先假设这三个参数存在初始值；
#     2）然后根据这三个参数求出前向和后向概率，这两个概率中存在这三个参数；
#     3）根据前向后向概率计算出新的参数值；
#     再根据新的参数值计算前向后向概率；

# 维特比算法：用来计算最可能的隐藏状态序列；
#     1）传统的动态规划是需要全量搜索，即把每一种可能性都考虑到，求出所有可能的路径的概率，选择概率最大的那个，但是这样的算法有非常大的运算量，不适合；
#     因此使用了动态代理的方法，维特比就是一个典型的动态代理方法；
#     2）维特比算法通过计算每个隐藏节点的状态最大概率P以及隐藏状态序列I来确定各个隐藏节点的最可能序列；
#     3）最大可能概率是指：上一个节点的最大可能概率P(n-1)、转移概率a、发射概率b三者结合的结果:max[P(n-1)*a]*b
#     4）上一个隐藏节点的最可能序列为：次节点的最大概率P与上一个节点到这个节点的转移概率：max[P(n-1)*a]
#     5）通过上面两步的计算可以回溯到最可能的隐藏序列为：
#     [第n个隐藏节点：最后一个节点的最大概率序列；第n-1个节点：最后一个节点的I值序列；第n-2个节点：第n-1个节点的I值序列；………………]





import numpy as np

class HMM:
    def __init__(self, Ann, Bnm, Pi, O):
        self.A = np.array(Ann, np.float)
        self.B = np.array(Bnm, np.float)
        self.Pi = np.array(Pi, np.float)
        self.O = np.array(O, np.float)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]

#  Viterbi算法作为类HMM的函数，相关代码如下

    def viterbi(self):
        # given O,lambda .finding I

        T = len(self.O)
        I = np.zeros(T, np.float)

        delta = np.zeros((T, self.N), np.float)  
        psi = np.zeros((T, self.N), np.float)

        for i in range(self.N):
            delta[0, i] = self.Pi[i] * self.B[i, self.O[0]]
            psi[0, i] = 0

        for t in range(1, T):
            for i in range(self.N):
                delta[t, i] = self.B[i,self.O[t]] * np.array( [delta[t-1,j] * self.A[j,i]
                    for j in range(self.N)] ).max() 
                psi[t,i] = np.array( [delta[t-1,j] * self.A[j,i] 
                    for j in range(self.N)] ).argmax()

        P_T = delta[T-1, :].max()
        I[T-1] = delta[T-1, :].argmax()

        for t in range(T-2, -1, -1):
            I[t] = psi[t+1, I[t+1]]

        return I

#   delta，psi 分别是 δ，ψ
#   其中np， 是import numpy as np， numpy这个包很好用，它的argmax()方法在这里非常实用。
#   3. 前向算法 ，后向算法，gamma－γ，xi－ξ，Baum_Welch算法及其python实现
#   HMM的公式推导包含很多概率值，如果你不能比较好地理解概率相关知识的话，相应的公式推导过程会比较难以理解，可以阅读Bishop写的《Pattern Recognition And Machine Learning》这本书，当然，在机器学习方面这本书一直都是经典。
#   前向（forward）概率矩阵alpha－α（公式书写时它根a很像，注意区分），
#   后向（backward）概率矩阵beta－β
#   算法定义和步骤参阅《统计学习方法》第175页或者文献二，
#   相关代码如下：

    def forward(self):
        T = len(self.O)
        alpha = np.zeros((T, self.N), np.float)

        for i in range(self.N):        
            alpha[0,i] = self.Pi[i] * self.B[i, self.O[0]]

        for t in range(T-1):
            for i in range(self.N):
                summation = 0   # for every i 'summation' should reset to '0'
                for j in range(self.N):
                    summation += alpha[t,j] * self.A[j,i]
                alpha[t+1, i] = summation * self.B[i, self.O[t+1]]

        summation = 0.0
        for i in range(self.N):
            summation += alpha[T-1, i]
        Polambda = summation
        return Polambda,alpha

    def backward(self):
        T = len(self.O)
        beta = np.zeros((T, self.N), np.float)
        for i in range(self.N):
            beta[T-1, i] = 1.0

        for t in range(T-2,-1,-1):
            for i in range(self.N):
                summation = 0.0     # for every i 'summation' should reset to '0'
                for j in range(self.N):
                    summation += self.A[i,j] * self.B[j, self.O[t+1]] * beta[t+1,j]
                beta[t,i] = summation

        Polambda = 0.0
        for i in range(self.N):
            Polambda += self.Pi[i] * self.B[i, self.O[0]] * beta[0, i]
        return Polambda, beta

#   Polambda表示P(O| λ)
#   接下来计算gamma－γ和 xi－ξ。 根据《统计学习方法》的公式可以得到如下代码:

    def compute_gamma(self,alpha,beta):
        T = len(self.O)
        gamma = np.zeros((T, self.N), np.float)       # the probability of Ot=q
        for t in range(T):
            for i in range(self.N):
                gamma[t, i] = alpha[t,i] * beta[t,i] / sum(
                    alpha[t,j] * beta[t,j] for j in range(self.N) )
        return gamma

    def compute_xi(self,alpha,beta):
        T = len(self.O)
        xi = np.zeros((T-1, self.N, self.N), np.float)  # note that: not T
        for t in range(T-1):   # note: not T
            for i in range(self.N):
                for j in range(self.N):
                    numerator = alpha[t,i] * self.A[i,j] * self.B[j,self.O[t+1]] * beta[t+1,j]
                    # the multiply term below should not be replaced by 'nummerator'，
                    # since the 'i,j' in 'numerator' are fixed.
                    # In addition, should not use 'i,j' below, to avoid error and confusion.
                    denominator = sum( sum(     
                        alpha[t,i1] * self.A[i1,j1] * self.B[j1,self.O[t+1]] * beta[t+1,j1] 
                        for j1 in range(self.N) )   # the second sum
                            for i1 in range(self.N) )    # the first sum
                    xi[t,i,j] = numerator / denominator
        return xi
 
#     注意计算时要传入参数alpha,beta
#     然后来实现Baum_Welch算法，根据《统计学习方法》或者文献二，
#     首先初始化参数，怎么初始化是很重要。因为Baum_Welch算法（亦是EM算法的一种特殊体现）并不能保证得到全局最优值，很容易就掉到局部最优然后出不来了。
#     当delta_lambda大于某一值时一直运行下去。
#     关于x的设置，如果过小，程序容易进入死循环，因为每一次的收敛过程lambda会有比较大的变化，那么当它接近局部／全局最优时，就会在左右徘徊一直是delta_lambda > x。

    def Baum_Welch(self):
        # given O list finding lambda model(can derive T form O list)
        # also given N, M, 
        T = len(self.O)
        V = [k for k in range(self.M)]

        # initialization - lambda 
        self.A = np.array(([[0,1,0,0],[0.4,0,0.6,0],[0,0.4,0,0.6],[0,0,0.5,0.5]]), np.float)
        self.B = np.array(([[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]]), np.float)

        # mean value may not be a good choice
        self.Pi = np.array(([1.0 / self.N] * self.N), np.float)  # must be 1.0 , if 1/3 will be 0
        # self.A = np.array([[1.0 / self.N] * self.N] * self.N) # must array back, then can use[i,j]
        # self.B = np.array([[1.0 / self.M] * self.M] * self.N)

        x = 1
        delta_lambda = x + 1
        times = 0
        # iteration - lambda
        while delta_lambda > x:  # x
            Polambda1, alpha = self.forward()           # get alpha
            Polambda2, beta = self.backward()            # get beta
            gamma = self.compute_gamma(alpha,beta)     # use alpha, beta
            xi = self.compute_xi(alpha,beta)

            lambda_n = [self.A,self.B,self.Pi]

            
            for i in range(self.N):
                for j in range(self.N):
                    numerator = sum(xi[t,i,j] for t in range(T-1))
                    denominator = sum(gamma[t,i] for t in range(T-1))
                    self.A[i, j] = numerator / denominator

            for j in range(self.N):
                for k in range(self.M):
                    numerator = sum(gamma[t,j] for t in range(T) if self.O[t] == V[k] )  # TBD
                    denominator = sum(gamma[t,j] for t in range(T))
                    self.B[i, k] = numerator / denominator

            for i in range(self.N):
                self.Pi[i] = gamma[0,i]

            # if sum directly, there will be positive and negative offset
            delta_A = map(abs, lambda_n[0] - self.A)  # delta_A is still a matrix
            delta_B = map(abs, lambda_n[1] - self.B)
            delta_Pi = map(abs, lambda_n[2] - self.Pi)
            delta_lambda = sum([ sum(sum(delta_A)), sum(sum(delta_B)), sum(delta_Pi) ])
            times += 1
            print(times)

        return self.A, self.B, self.Pi

#     4.带scale的Baum-Welch算法，多项观测序列带scale的Baum-Welch算法
#     理论上来说上面已经完整地用代码实现了HMM, 然而事实总是没有那么简单，后续还有不少问题需要解决，不过这篇文章只提两点，一个是用scale解决计算过程中容易发送的浮点数下溢问题，另一个是同时输入多个观测序列的改进版Baum-Welch算法。
#     参考文献二： scaling problem
#     根据文献二的公式我们加入scale，重写forward(),backward(),Baum-Welch() 三个方法。
    def forward_with_scale(self):
        T = len(self.O)
        alpha_raw = np.zeros((T, self.N), np.float)
        alpha = np.zeros((T, self.N), np.float)
        c = [i for i in range(T)]  # scaling factor; 0 or sequence doesn't matter

        for i in range(self.N):        
            alpha_raw[0,i] = self.Pi[i] * self.B[i, self.O[0]]
            
        c[0] = 1.0 / sum(alpha_raw[0,i] for i in range(self.N))
        for i in range(self.N):
            alpha[0, i] = c[0] * alpha_raw[0,i]

        for t in range(T-1):
            for i in range(self.N):
                summation = 0.0
                for j in range(self.N):
                    summation += alpha[t,j] * self.A[j, i]
                alpha_raw[t+1, i] = summation * self.B[i, self.O[t+1]]

            c[t+1] = 1.0 / sum(alpha_raw[t+1,i1] for i1 in range(self.N))
            
            for i in range(self.N):
                alpha[t+1, i] = c[t+1] * alpha_raw[t+1, i]
        return alpha, c


    def backward_with_scale(self,c):
        T = len(self.O)
        beta_raw = np.zeros((T, self.N), np.float)
        beta = np.zeros((T, self.N), np.float)
        for i in range(self.N):
            beta_raw[T-1, i] = 1.0
            beta[T-1, i] = c[T-1] * beta_raw[T-1, i]

        for t in range(T-2,-1,-1):
            for i in range(self.N):
                summation = 0.0
                for j in range(self.N):
                    summation += self.A[i,j] * self.B[j, self.O[t+1]] * beta[t+1,j]
                beta[t,i] = c[t] * summation   # summation = beta_raw[t,i]
        return beta

    def Baum_Welch_with_scale(self):
        T = len(self.O)
        V = [k for k in range(self.M)]

        # initialization - lambda   ,  should be float(need .0)
        self.A = np.array([[0.2,0.2,0.3,0.3],[0.2,0.1,0.6,0.1],[0.3,0.4,0.1,0.2],[0.3,0.2,0.2,0.3]])
        self.B = np.array([[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]])

        x = 5
        delta_lambda = x + 1
        times = 0
        # iteration - lambda
        while delta_lambda > x:  # x
            alpha,c = self.forward_with_scale()
            beta = self.backward_with_scale(c)    

            lambda_n = [self.A,self.B,self.Pi]
            
            for i in range(self.N):
                for j in range(self.N):
                    numerator_A = sum(alpha[t,i] * self.A[i,j] * self.B[j, self.O[t+1]]
                                 * beta[t+1,j] for t in range(T-1))
                    denominator_A = sum(alpha[t,i] * beta[t,i] / c[t] for t in range(T-1))
                    self.A[i, j] = numerator_A / denominator_A

            for j in range(self.N):
                for k in range(self.M):
                    numerator_B = sum(alpha[t,j] * beta[t,j] / c[t]
                                for t in range(T) if self.O[t] == V[k] )  # TBD
                    denominator_B = sum(alpha[t,j] * beta[t,j] / c[t] for t in range(T))
                    self.B[j, k] = numerator_B / denominator_B

            # Pi have no business with c
            denominator_Pi = sum(alpha[0,j] * beta[0,j] for j in range(self.N))
            for i in range(self.N):      
                self.Pi[i] = alpha[0,i] * beta[0,i] / denominator_Pi 
                #self.Pi[i] = gamma[0,i]   

            # if sum directly, there will be positive and negative offset
            delta_A = map(abs, lambda_n[0] - self.A)  # delta_A is still a matrix
            delta_B = map(abs, lambda_n[1] - self.B)
            delta_Pi = map(abs, lambda_n[2] - self.Pi)
            delta_lambda = sum([ sum(sum(delta_A)), sum(sum(delta_B)), sum(delta_Pi) ])

            times += 1
            print(times)

        return self.A, self.B, self.Pi

#   第二个问题，根据文献二，我们直接实现带scale的修改版Baum-Welch算法。为了方便，我们将这个函数单独出来，写在HMM类的外面:

# for multiple sequences of observations symbols(with scaling alpha & beta)
# out of class HMM, independent function
def modified_Baum_Welch_with_scale(O_set):
    # initialization - lambda  
    A = np.array([[0.2,0.2,0.3,0.3],[0.2,0.1,0.6,0.1],[0.3,0.4,0.1,0.2],[0.3,0.2,0.2,0.3]])
    B = np.array([[0.2,0.2,0.3,0.3],[0.2,0.1,0.6,0.1],[0.3,0.4,0.1,0.2],[0.3,0.2,0.2,0.3]])
    # B = np.array([[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]])
    Pi = [0.25,0.25,0.25,0.25]                               

    # computing alpha_set, beta_set, c_set
    O_length = len(O_set)
    whatever = [j for j in range(O_length)]
    alpha_set, beta_set = whatever, whatever
    c_set = [j for j in range(O_length)]  # can't use whatever, the c_set will be 3d-array ???

    N = A.shape[0]
    M = B.shape[1]
    T = [j for j in range(O_length)]   # can't use whatever, the beta_set will be 1d-array ???
    for i in range(O_length):
        T[i] = len(O_set[i])
    V = [k for k in range(M)]  
    
    x = 1
    delta_lambda = x + 1
    times = 0    
    while delta_lambda > x:   # iteration - lambda        
        lambda_n = [A, B]
        for i in range(O_length):
            alpha_set[i], c_set[i] = HMM(A, B, Pi, O_set[i]).forward_with_scale()
            beta_set[i] = HMM(A, B, Pi, O_set[i]).backward_with_scale(c_set[i])

        for i in range(N): 
            for j in range(N):

                numerator_A = 0.0
                denominator_A = 0.0
                for l in range(O_length):
                    
                    raw_numerator_A = sum( alpha_set[l][t,i] * A[i,j] * B[j, O_set[l][t+1]] 
                                * beta_set[l][t+1,j] for t in range(T[l]-1) )
                    numerator_A += raw_numerator_A

                    raw_denominator_A = sum( alpha_set[l][t,i] * beta_set[l][t,i] / c_set[l][t]
                                 for t in range(T[l]-1) )                    
                    denominator_A += raw_denominator_A
                         
                A[i, j] = numerator_A / denominator_A

        for j in range(N):
            for k in range(M):

                numerator_B = 0.0
                denominator_B = 0.0
                for l in range(O_length):
                    raw_numerator_B = sum( alpha_set[l][t,j] * beta_set[l][t,j] 
                                / c_set[l][t] for t in range(T[l]) if O_set[l][t] == V[k] )
                    numerator_B += raw_numerator_B

                    raw_denominator_B = sum( alpha_set[l][t,j] * beta_set[l][t,j] 
                                / c_set[l][t] for t in range(T[l]) )
                    denominator_B += raw_denominator_B    
                B[j, k] = numerator_B / denominator_B

        # Pi should not need to computing in this case, 
        # in other cases, will get some corresponding Pi

        # if sum directly, there will be positive and negative offset
        delta_A = map(abs, lambda_n[0] - A)  # delta_A is still a matrix
        delta_B = map(abs, lambda_n[1] - A)
        delta_lambda = sum([ sum(sum(delta_A)), sum(sum(delta_B)) ])

        times += 1
        print(times)

    return A, B

