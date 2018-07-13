# --*-- coding:utf-8 --*--

# BDT的运算：
# 1）使用最小二乘法或者最小熵模型来形成一个原始的决策树；
# 2）计算各个叶子节点中的点与此节点上均值的残差，得到一个新的残差矩阵；
# 3）同样的方法构建残差的决策树，新的决策树应该是与原有的决策树的结构一致；
# 4）将形成的决策树按照统一位置的节点相加（是指每个节点的输出值相加）；
# 5）得出的结果即为各个数据的具体值；

# GBDT的运算:
# 基于残差的思路进行迭代：
# 1）对于决策树（离散）的样式，可以先根据原有的数据依据信息增益等方式，构建一个决策树；
#        对于回归树（连续）的样式，可以根据最小二乘法原则，构建一个决策树；
#        如果有一个模型f(x)，可以计算残差，构建一个原始模型，然后筒骨弓残差迭代；
# 2）计算各个叶子节点中的点与此节点上均值的残差，得到一个新的残差矩阵；
# 3）将这个残差矩阵加到原始决策树 上，然后计算损失函数；
# 4）如果损失函数满足最优，即最小值，迭代停止；
# 5）如果损失函数不满足最优，则再计算新决策树的残差，然后再加和，循环上面的步骤；

# 基于boosting的思路进行迭代：
# 1）如果构建一个f(x)的模型，假设这个模型由k个基模型构成，形成一个加法算式：
#    yi=sum(f(x))
#    yi=y(i-1)+f(x)
# 2)这个模型y对应的损失函数为（即每一次迭代所产生的损失函数l(y,f(x))的叠加）：
#    L=sum(l(yi,f(x)))
# 3)如果考虑偏差和方差，则公式可以变换为：
#    obj=sum(l(yi,f(x)))+sum(复杂度)
#    obj=sum(l(yi,y(i-1)+f(x)))+sum(复杂度)
# 4)将上面的公式与泰勒公式结合，形成新的目标函数：
#    f(x+delta(x))=f(x)+f_dao(x)*delta(x)+(1/2)(f_shuangdao(x)*delta(x)^2)
# 5)结合之后模型的目标函数为：
#    gi是损失函数的一阶导；hi是损失函数的二阶导；损失函数为y(t-1)
#    obj=sum(gi*f(x)+(1/2)*hi*f(x)^2)+(复杂度)
   
# 1)如果建一个决策树模型，则将f(x)转换成w，将惩罚系数添加L2惩罚系数；
# 2）通过上面的公式可以得到每个叶子节点的取值；
#     w*=-Gj/(Hj+lambda)
# 3)计算损失函数在每个样本上的一阶导和二阶导，即gi与hi；
# 4)算法在拟合的每一步都生成一颗决策树；

# 对于单棵树：
# 1）枚举所有可能的树结构，即q；
# 2）计算每种树结构的目标函数值，即损失函数obj：
# obj=-(1/2)*sum(Gj^2/(Hj+lambda)))+惩罚系数；
# 3)取目标函数最小的值为最佳的树结构，根据等式w*=-Gj/(Hj+lambda)计算每个叶子节点的w值，即样本的预测值；

# 对于多棵树（贪心算法）：生成树的过程中，通过用损失函数来计算收益；
# 1）从深度为零的树开始，对每个叶节点枚举所有的可用特征；
# 2）针对每个特征，把属于该节点的训练样本根据该特征值升序排列，通过线性扫描的方式来决定该特征的最佳分裂点，并记录该特征的最大收益；
#   计算收益的方式为：
#     分列前的目标函数为：lossqian=-(1/2)*[(GL+GR)^2/(HL+HR+lambda)]+r；
#     分裂后的目标函数为：losshou =-(1/2)*[GL^2/(HL+lambda)+GR^2/(HR+lambda)]+2r
#     则分裂后的收益为：    gain=lossqian-losshou
# 3）选择收益最大的特征作为分裂特征，用该特征的最佳分裂点作为分裂位置，把该节点生长出左右两个新的叶节点，并为每个新节点关联对应的样本集；
# 4）循环上面三步，直到满足特定的条件为止；
# 5）整个过程生成一棵树，然后计算每个叶子节点的Gj和Hj，并计算出w；
# 6）把新生成的决策树f(x)加入到yi=y(i-1)+a(f(x)),a是用来抑制过拟合的；



import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
gbdt=GradientBoostingRegressor(
  loss='ls', learning_rate=0.1, n_estimators=100, subsample=1, min_samples_split=2, min_samples_leaf=1, 
  max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, 
  warm_start=False
)
train_feat=np.genfromtxt("train_feat.txt",dtype=np.float32)
train_id=np.genfromtxt("train_id.txt",dtype=np.float32)
test_feat=np.genfromtxt("test_feat.txt",dtype=np.float32)
test_id=np.genfromtxt("test_id.txt",dtype=np.float32)
print(train_feat.shape，rain_id.shape，est_feat.shape，est_id.shape)
gbdt.fit(train_feat,train_id)
pred=gbdt.predict(test_feat)
total_err=0
for i in range(pred.shape[0]):
    print(pred[i],test_id[i])
    err=(pred[i]-test_id[i])/test_id[i]
    total_err+=err*err
print(total_err/pred.shape[0])






















