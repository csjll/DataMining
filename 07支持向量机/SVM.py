# --*-- coding:utf-8 --*--

# SVM主要解决的是非线性的分类问题：
# 1、现实生活中遇到多维度的点，这些点无法直接通过一个超平面将其完全的分开，需要将这些数据点投射到较高的维度，实现高纬度分类，
#     分类的动作是在高纬度，但是超平面参数的运算在低纬度（通过核函数进行转换）；
# 2、具体的做法为：
#     1）假设数据之间存在一个超平面将各个数据点完整的分开；
#     2）计算每个点到这个超平面的距离（函数间隔）；
#     3）距离最大时对应的超平面即为用来分类的超平面；
#     4）此时问题转变为计算最大的函数间隔，即计算f(w)=min(||w||^2);
#     5）由于这个公式存在一定的约束条件，则需要使用拉格朗日乘子法来运算；
#     L(w,b,a)=(||w||^2)+a*g(w,b);
#     6）计算其min(max L(w,b,a))即可求得超平面的最优参数;
# 3、min(max L(w,b,a))的计算过程中需要用到对偶运算、求偏导、SMO算法、核函数等技巧；
#     1）对偶运算将minmax转化为maxmin以方面求取对应值；
#     2）通过求min偏导得到w与b的解；同时通过max求出a的值；
#     3）将w,b代入到f(w,b)中，将f(w,b)转化为<xi,x>内积的函数；
#     4）此时需要用核函数来运算分类超平面；
# 4、通过上面的运算得到f(w,b)的分类器公式，这个公式通过计算x的内积来确定新进入点的分类问题，这样的
#   方法只限于在线性问题中，但是当遇到非线性问题时，x的内积就不能发挥作用，此时需要用到核函数，来将
#   x转化成更高维度，使其变成线性可分的数据，同时，用核函数计算出这个更高维度的x的内积，这样的转换
#   完成了非线性向线性的转换；
# 5、非线性：f(w,b)=sum(a*yi*<xi,x>)+b
#   线性：f(w,b)=sum(a*yi*K<xi,x>)+b
#   两者的运算值不一样，但是由于svm用来进行分类，这样的运算不影响点的分类，所以可以使用；
#   K<xi,x>的作用有两个：
#      1）将低维的x转化成高维；x——>g(x)
#      2）用低维的运算计算高维的内积；<g(xi),g(x)>——>K<xi,x>
# 6、对于outlier点，需要添加惩罚因子(松弛变量)e：
#      1）在约束条件中添加1-e；
#      2）在距离公式中添加C*sum(e)；
#   以上两者结合成拉格朗日函数会得到新的L(w,b,a,r,e)
#   运算结果对w和核函数没有影响，只影响到约束条件；

# 总结：
# 1）通过计算最大间隔f(w)来确定分类超平面；
# 2）用拉格朗日乘子法来计算w与b；
# 3）用核函数来计算高维的参数；

# 合页损失函数：取正值的函数；

# SMO算法，求取（a1,a2,a3,a4,a5,a6,a7……）：
# 获得对偶问题的二次规划公式；
# 1）选取其中的两个参数（ai,aj）;
# 2）固定其他的参数，将（ai,aj）代入二次规划公式进行求解，更新（ai,aj）
# 循环上面两步，直至参数收敛；


"""
Created on Tue Nov 22 11:24:22 2016
@author: Administrator
"""

# Mathieu Blondel, September 2010
# License: BSD 3 clause

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        '''
         数组的flatten和ravel方法将数组变为一个一维向量（铺平数组）。
         flatten方法总是返回一个拷贝后的副本，
         而ravel方法只有当有必要时才返回一个拷贝后的副本（所以该方法要快得多，尤其是在大数组上进行操作时）
       '''
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        '''
        这里a>1e-5就将其视为非零
        '''
        sv = a > 1e-5     # return a list with bool values
        ind = np.arange(len(a))[sv]  # sv's index
        self.a = a[sv]
        self.sv = X[sv]  # sv's data
        self.sv_y = y[sv]  # sv's labels
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        '''
        这里相当于对所有的支持向量求得的b取平均值
        '''
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                # linear_kernel相当于在原空间，故计算w不用映射到feature space
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        # w有值，即kernel function 是 linear_kernel，直接计算即可
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        # w is None --> 不是linear_kernel,w要重新计算
        # 这里没有去计算新的w（非线性情况不用计算w）,直接用kernel matrix计算预测结果
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    import pylab as pl

    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    # 仅仅在Linears使用此函数作图，即w存在时
    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

    def plot_contour(X1_train, X2_train, clf):
        # 作training sample数据点的图
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        # 做support vectors 的图
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")
        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        # pl.contour做等值线图
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_soft():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=0.1)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    # test_soft()
    # test_linear()
    test_non_linear()





















# svm的一些常用方法：
# svc使用代码示例（我演示的是最简单的，官网上还有很多看起来很漂亮的分类示例，感兴趣的可以自己参考下）：
'''
SVC参数解释 
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0； 
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF"; 
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂； 
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features; 
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效； 
（6）probablity: 可能性估计是否使用(true or false)； 
（7）shrinking：是否进行启发式； 
（8）tol（default = 1e - 3）: svm结束标准的精度; 
（9）cache_size: 制定训练所需要的内存（以MB为单位）； 
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应； 
（11）verbose: 跟多线程有关，不大明白啥意思具体； 
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited; 
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None 
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。 
 ps：7,8,9一般不考虑。 
'''  
from sklearn.svm import SVC  
import numpy as np 
 
X = np.array([[-1,-1],[-2,-1],[1,1],[2,1]])  
y = np.array([1,1,2,2])  
  
clf = SVC()  
clf.fit(X,y)  
print(clf.fit(X,y))  
print(clf.predict([[-0.8,-1]]))
# 输出结果为：
# 第一个打印出的是svc训练函数的参数，其更多参数说明请参考：点击阅读
# 最后一行打印的是预测结果
# NuSVC（Nu-Support Vector Classification.）：核支持向量分类，和SVC类似，也是基于libsvm实现的，但不同的是通过一个参数空值支持向量的个数
# 示例代码：
''' 
NuSVC参数 
nu：训练误差的一个上界和支持向量的分数的下界。应在间隔（0，1 ]。 
其余同SVC 
'''  
import numpy as np  
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  
y = np.array([1, 1, 2, 2])  
from sklearn.svm import NuSVC  
clf = NuSVC()  
clf.fit(X, y)   
print(clf.fit(X,y)) 
print(clf.predict([[-0.8, -1]]))  


# LinearSVC（Linear Support Vector Classification）：线性支持向量分类，类似于SVC，但是其使用的核函数是”linear“
# 上边介绍的两种是按照brf（径向基函数计算的，其实现也不是基于LIBSVM，所以它具有更大的灵活性在选择处罚和损失函数时，而且可以适应更大的数
# 据集，他支持密集和稀疏的输入是通过一对一的方式解决的
# 代码使用实例如下：
'''  
LinearSVC 参数解释  
C：目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；  
loss ：指定损失函数  
penalty ：  
dual ：选择算法来解决对偶或原始优化问题。当n_samples > n_features 时dual=false。  
tol ：（default = 1e - 3）: svm结束标准的精度;  
multi_class：如果y输出类别包含多类，用来确定多类策略， ovr表示一对多，“crammer_singer”优化所有类别的一个共同的目标  
如果选择“crammer_singer”，损失、惩罚和优化将会被被忽略。  
fit_intercept ：  
intercept_scaling ：  
class_weight ：对于每一个类别i设置惩罚系数C = class_weight[i]*C,如果不给出，权重自动调整为 n_samples / (n_classes * np.bincount(y))  
verbose：跟多线程有关，不大明白啥意思具体<pre name="code" class="python">  
'''  
from sklearn.svm import SVC  
  
X=[[0],[1],[2],[3]]  
Y = [0,1,2,3]  
  
clf = SVC(decision_function_shape='ovo') #ovo为一对一  
clf.fit(X,Y)  
print(clf.fit(X,Y))  
  
dec = clf.decision_function([[1]])    #返回的是样本距离超平面的距离  
print(dec)
  
clf.decision_function_shape = "ovr"  
dec =clf.decision_function([1]) #返回的是样本距离超平面的距离  
print(dec)
  
#预测  
print(clf.predict([1]))

# random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。max_iter ：

import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X, y) 
print(clf.fit(X,y))
print(clf.predict([[-0.8, -1]])) 

# 结果如下：
# 更多关于LinearSVC请参考：点击阅读
# 在CODE上查看代码片派生到我的代码片
''' 
Created on 2016年4月29日 
@author: Gamer Think 
'''  
  
from sklearn.svm import SVC,LinearSVC  
  
X=[[0],[1],[2],[3]]  
Y = [0,1,2,3]  
  
''''' 
SVC and NuSVC 
'''  
clf = SVC(decision_function_shape='ovo') #ovo为一对一  
clf.fit(X,Y)  
print("SVC:",clf.fit(X,Y))
  
dec = clf.decision_function([[1]])    #返回的是样本距离超平面的距离  
print("SVC:",dec)
  
clf.decision_function_shape = "ovr"  
dec =clf.decision_function([1]) #返回的是样本距离超平面的距离  
print("SVC:",dec)
  
#预测  
print("预测：",clf.predict([1])) 
 
lin_clf = LinearSVC()
lin_clf.fit(X, Y) 
dec = lin_clf.decision_function([[1]])
print("LinearSVC:",dec.shape[1])

# Unbalanced problems（数据不平衡问题）
# 对于非平衡级分类超平面，使用不平衡SVC找出最优分类超平面，基本的思想是，我们先找到一个普通的分类超平面，自动进行校正，求出最优的分类超平面
# 这里可以使用 SVC(kernel="linear")
# 针对下面的svc可以使用 clf=SGDClassifier(n_iter=100,alpha=0.01)
''' 
Created on 2016年5月4日 
@author: Gamer Think 
'''  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import svm  
#from sklearn.linear_model import SGDClassifier  
# we create 40 separable points  
rng = np.random.RandomState(0)  
n_samples_1 = 1000  
n_samples_2 = 100  
X = np.r_[1.5 * rng.randn(n_samples_1, 2),0.5 * rng.randn(n_samples_2, 2) + [2, 2]]  
y = [0] * (n_samples_1) + [1] * (n_samples_2)  
print(X)  
print(y)  
  
# fit the model and get the separating hyperplane  
clf = svm.SVC(kernel='linear', C=1.0)  
clf.fit(X, y)  
  
w = clf.coef_[0]  
a = -w[0] / w[1]      #a可以理解为斜率  
xx = np.linspace(-5, 5)  
yy = a * xx - clf.intercept_[0] / w[1]    #二维坐标下的直线方程  
  
  
# get the separating hyperplane using weighted classes  
wclf = svm.SVC(kernel='linear', class_weight={1: 10})  
wclf.fit(X, y)  
  
ww = wclf.coef_[0]  
wa = -ww[0] / ww[1]  
wyy = wa * xx - wclf.intercept_[0] / ww[1]   #带权重的直线  
  
# plot separating hyperplanes and samples  
h0 = plt.plot(xx, yy, 'k-', label='no weights')  
h1 = plt.plot(xx, wyy, 'k--', label='with weights')  
plt.scatter(X[:, 0], X[:, 1], c=y)  
plt.legend()  
  
plt.axis('tight')  
plt.show()

# 2：Regression
# 支持分类的支持向量机可以推广到解决回归问题，这种方法称为支持向量回归
# 支持向量分类所产生的模型仅仅依赖于训练数据的一个子集，因为构建模型的成本函数不关心在超出边界范围的点，类似的，通过支持向量回归产生的模型依赖于训练数据的一个子集，因
# 为构建模型的函数忽略了靠近预测模型的数据集。
# 有三种不同的实现方式：
#    支持向量回归SVR，nusvr和linearsvr。
# linearsvr提供了比SVR更快实施但只考虑线性核函数，而nusvr实现比SVR和linearsvr略有不同。
# 作为分类类别，训练函数将X,y作为向量，在这种情况下y是浮点数

from sklearn import svm  
X = [[0, 0], [2, 2]]  
y = [0.5, 2.5]  
clf = svm.SVR()  
clf.fit(X, y)   
svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',  
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)  
clf.predict([[1, 1]])  
np.array([ 1.5])

# 下面看一个使用SVR做线性回归的例子：
''' 
Created on 2016年5月4日 
@author: Gamer Think 
'''  
import numpy as np  
from sklearn.svm import SVR  
import matplotlib.pyplot as plt  
  
###############################################################################  
# Generate sample data  
X = np.sort(5 * np.random.rand(40, 1), axis=0)  #产生40组数据，每组一个数据，axis=0决定按列排列，=1表示行排列  
y = np.sin(X).ravel()   #np.sin()输出的是列，和X对应，ravel表示转换成行  
  
###############################################################################  
# Add noise to targets  
y[::5] += 3 * (0.5 - np.random.rand(8))  
  
###############################################################################  
# Fit regression model  
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  
svr_lin = SVR(kernel='linear', C=1e3)  
svr_poly = SVR(kernel='poly', C=1e3, degree=2)  
y_rbf = svr_rbf.fit(X, y).predict(X)  
y_lin = svr_lin.fit(X, y).predict(X)  
y_poly = svr_poly.fit(X, y).predict(X)  
  
###############################################################################  
# look at the results  
lw = 2  
plt.scatter(X, y, color='darkorange', label='data')  
plt.hold('on')  
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')  
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')  
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')  
plt.xlabel('data')  
plt.ylabel('target')  
plt.title('Support Vector Regression')  
plt.legend()  
plt.show()


