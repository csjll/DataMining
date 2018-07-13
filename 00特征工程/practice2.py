#-*- coding:utf-8-*-
from sklearn.datasets import load_iris
import numpy as np
from numpy import vstack, array, NaN
from sklearn.preprocessing.data import StandardScaler, MinMaxScaler, Normalizer, Binarizer, OneHotEncoder,\
    PolynomialFeatures
from sklearn.preprocessing import Imputer   
from sklearn.preprocessing._function_transformer import FunctionTransformer
from math import log1p
from sklearn.feature_selection.variance_threshold import VarianceThreshold
from sklearn.feature_selection.univariate_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
# from minepy import MINE
from tensorflow.contrib import estimator
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection.from_model import SelectFromModel
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.decomposition import PCA
# from sklearn.lda import LDA
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search

iris=load_iris()

# 花萼长度、花萼宽度、花瓣长度、花瓣宽度
print(iris.data) 
# 0：山鳶花；1：杂色鸢花；2：维吉尼亚鸢尾
print(iris.target)


# 数据预处理：
# 1) 无量纲化：
#     a.标准化；
#     b.区间缩放法（归一化）；
#     c.正则化；
#     d.数据格式化；
# 2) 对定量特征二值化；
# 3) 对定性特征哑编码；
# 4) 缺失值计算；
# 5) 数据变换；
# 6) 数据清洗：
#     a.简单的impossible数据；
#     b.组合或统计属性判断；
#     c.补齐对应的缺失值； 
# 7）数据采样(如果数据不均衡)：
#     a.上采样；
#     b.下采样；

# 标准化：将同一列的数据按比例缩放；
# 正则化：将数据转化成正则形式；
# print(StandardScaler().fit_transform(iris.data))

# 区间缩放法：
# print(MinMaxScaler().fit_transform(iris.data))

# 归一化：将不同维度的数据映射到0-1之间，去除量纲；
# print(Normalizer().fit_transform(iris.data))

# 对数据二值化：
# print(Binarizer(threshold=3).fit_transform(iris.data))

# 对数据进行哑编码（类别型），主要是对离散性的不能进行正常运算的数据进行运算：
# print(OneHotEncoder().fit_transform((iris.target).reshape(-1, 1)))
# print((iris.target).reshape(-1, 1))

# 缺失值计算：默认将缺失值定为mean；
# print(Imputer().fit_transform(vstack((array([NaN, NaN, NaN, NaN]), iris.data))))

# 多项式转换：多项式转换
# print(PolynomialFeatures().fit_transform(iris.data))

# 对数转换：
# print(FunctionTransformer(log1p).fit_transform(iris.data))


# 特征选择：
# 1）过滤法：
#   a.方差选择;
#   b.相关系数法;
#   c.卡方检验;
#   d.互信息法;
# 2）包装法：
#   a.递归特征消除法;
# 3）嵌入法：
#   a.基于惩罚项的特征选择;
#   b.基于GBDT的特征选择;

# 方差选择法(此处选择出第三列为常用特征)：
# print(VarianceThreshold(threshold=3).fit_transform(iris.data))

# 相关系数法（pearsonr相关系数是指自变量与因变量之间的相关性）：
# print(SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target))
# lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T

# 相当于（即计算xy的相关性，x为一个矩阵，）：
# def func2(X, Y):
#     def func(x):
#         return pearsonr(x, Y)
#     z=func(X)
#     t=array(z, X.T).T
#     return t
# 
# a=[1,2,3]
# b=[0,1,2]
# print(pearsonr(a, b))

# 卡方检验(通过计算自变量与因变量频数的变化来计算相关性)：
# print(SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target))

# 互信息是指一个事件的出现对另一个事件的出现所贡献的信息量;
# 互信息是计算x中的各个元素与y中的元素相互影响的信息量。
# def mic(x, y):
#     m=MINE()
#     m.compute_score(x, y)
#     return (m.mic(), 0.5)
#     
# print(SelectKBest(lambda X, Y:array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target))

# 递归特征消除法(用逻辑回归函数来进行训练，确定2个特征)：
# print(RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target))

# 基于惩罚项的特征选择(保留多个对目标值有同等相关性的特征中的一个)：
# print(SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target))
# 没选到的不代表不重要，因此需要结合l2惩罚因子来进行优化

# 基于树模型的特征选择法(通过GBDT来进行权值系数的计算，然后再根据权值系数进行选择)：
# SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)

# 降维：
# 1）主成分分析法（PCA）；
# print(PCA(n_components=2).fit_transform(iris.data))
# 2）线性判别分析法（LDA）；
# print(LDA(n_components=2).fit_transform(iris.data, iris.target))
# 3）奇异值分解（SVD）；

#数据挖掘的关键：
# 1）并行处理；
#     a.整体并行处理；
#     b.部分并行处理；
# 2）流水线处理；
# 3）自动化调参；
# 4）持久化；

# 整体并行处理：
# 新建将整体特征矩阵进行对数函数转换的对象；
# step2_1=('TopLog', FunctionTransformer(log1p))
# 新建将整体特征矩阵进行二值化类的对象；
# step2_2=('ToBinary', Binarizer())
# transformer_list即为需要并行处理的对象列表
# step2=('FeatureUnion', FeatureUnion(transformer_list=[step2_1, step2_2]))

# 部分并行处理：
# 即提取部分列作为特征选择的并行处理：
# 新建将整体特征矩阵进行对数函数转换的对象；
# step2_1=('TopLog', FunctionTransformer(log1p))
# 新建将整体特征矩阵进行二值化类的对象；
# step2_2=('ToBinary', Binarizer())
# FeatureUnionExt需要重构的。
# step2=('FeatureUnionExt', FeatureUnionExt(tranfromer_list=[step2_1, step2_2, step2_3], idx_list=[[0],[1,2,3],[4]]))

# 流水线处理（即上一个流程的输出是下一个流程的输入）：
#新建计算缺失值的对象
# step1 = ('Imputer', Imputer())
#新建将部分特征矩阵进行定性特征编码的对象
# step2_1 = ('OneHotEncoder', OneHotEncoder(sparse=False))
#新建将部分特征矩阵进行对数函数转换的对象
# step2_2 = ('ToLog', FunctionTransformer(log1p))
#新建将部分特征矩阵进行二值化类的对象
# step2_3 = ('ToBinary', Binarizer())
#新建部分并行处理对象，返回值为每个并行工作的输出的合并
# step2 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=[step2_1, step2_2, step2_3], idx_list=[[0], [1, 2, 3], [4]]))
#新建无量纲化对象
# step3 = ('MinMaxScaler', MinMaxScaler())
#新建卡方校验选择特征的对象
# step4 = ('SelectKBest', SelectKBest(chi2, k=3))
#新建PCA降维的对象
# step5 = ('PCA', PCA(n_components=2))
#新建逻辑回归的对象，其为待训练的模型作为流水线的最后一步
# step6 = ('LogisticRegression', LogisticRegression(penalty='l2'))
#新建流水线处理对象
#参数steps为需要流水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
# pipeline = Pipeline(steps=[step1, step2, step3, step4, step5, step6])

# 自动化调参(新建网格搜索对象，第一参数为待训练的模型，param_grid为待调参数组成的网格)：
# grid_search=GridSearchCV(Pipeline, param_grid={'FeatureUnionExt_ToBinary_threshold':[1.0, 2.0, 3.0, 4.0], 'LogisticRegression_C':[0.1, 0.2, 0.4, 0.8]})
# 训练参数
# grid_search.fit(iris.data, iris.target)

# 持久化：
# from externals.joblib import dump, load
# 第一个参数为内存中的对象，第二个参数为保存的名称，第三个参数为压缩级别；
# dump(grid_search, 'grid_search.dmp', compress=3)
# 从文件系统中加载数据到内存。
# grid_search=load('grid_search.dmp')





















































