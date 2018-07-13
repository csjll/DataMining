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

# ���೤�ȡ������ȡ����곤�ȡ�������
print(iris.data) 
# 0��ɽ�S����1����ɫ𰻨��2��ά�������β
print(iris.target)


# ����Ԥ����
# 1) �����ٻ���
#     a.��׼����
#     b.�������ŷ�����һ������
#     c.���򻯣�
#     d.���ݸ�ʽ����
# 2) �Զ���������ֵ����
# 3) �Զ��������Ʊ��룻
# 4) ȱʧֵ���㣻
# 5) ���ݱ任��
# 6) ������ϴ��
#     a.�򵥵�impossible���ݣ�
#     b.��ϻ�ͳ�������жϣ�
#     c.�����Ӧ��ȱʧֵ�� 
# 7�����ݲ���(������ݲ�����)��
#     a.�ϲ�����
#     b.�²�����

# ��׼������ͬһ�е����ݰ��������ţ�
# ���򻯣�������ת����������ʽ��
# print(StandardScaler().fit_transform(iris.data))

# �������ŷ���
# print(MinMaxScaler().fit_transform(iris.data))

# ��һ��������ͬά�ȵ�����ӳ�䵽0-1֮�䣬ȥ�����٣�
# print(Normalizer().fit_transform(iris.data))

# �����ݶ�ֵ����
# print(Binarizer(threshold=3).fit_transform(iris.data))

# �����ݽ����Ʊ��루����ͣ�����Ҫ�Ƕ���ɢ�ԵĲ��ܽ���������������ݽ������㣺
# print(OneHotEncoder().fit_transform((iris.target).reshape(-1, 1)))
# print((iris.target).reshape(-1, 1))

# ȱʧֵ���㣺Ĭ�Ͻ�ȱʧֵ��Ϊmean��
# print(Imputer().fit_transform(vstack((array([NaN, NaN, NaN, NaN]), iris.data))))

# ����ʽת��������ʽת��
# print(PolynomialFeatures().fit_transform(iris.data))

# ����ת����
# print(FunctionTransformer(log1p).fit_transform(iris.data))


# ����ѡ��
# 1�����˷���
#   a.����ѡ��;
#   b.���ϵ����;
#   c.��������;
#   d.����Ϣ��;
# 2����װ����
#   a.�ݹ�����������;
# 3��Ƕ�뷨��
#   a.���ڳͷ��������ѡ��;
#   b.����GBDT������ѡ��;

# ����ѡ��(�˴�ѡ���������Ϊ��������)��
# print(VarianceThreshold(threshold=3).fit_transform(iris.data))

# ���ϵ������pearsonr���ϵ����ָ�Ա����������֮�������ԣ���
# print(SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target))
# lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T

# �൱�ڣ�������xy������ԣ�xΪһ�����󣬣���
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

# ��������(ͨ�������Ա����������Ƶ���ı仯�����������)��
# print(SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target))

# ����Ϣ��ָһ���¼��ĳ��ֶ���һ���¼��ĳ��������׵���Ϣ��;
# ����Ϣ�Ǽ���x�еĸ���Ԫ����y�е�Ԫ���໥Ӱ�����Ϣ����
# def mic(x, y):
#     m=MINE()
#     m.compute_score(x, y)
#     return (m.mic(), 0.5)
#     
# print(SelectKBest(lambda X, Y:array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target))

# �ݹ�����������(���߼��ع麯��������ѵ����ȷ��2������)��
# print(RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target))

# ���ڳͷ��������ѡ��(���������Ŀ��ֵ��ͬ������Ե������е�һ��)��
# print(SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target))
# ûѡ���Ĳ�������Ҫ�������Ҫ���l2�ͷ������������Ż�

# ������ģ�͵�����ѡ��(ͨ��GBDT������Ȩֵϵ���ļ��㣬Ȼ���ٸ���Ȩֵϵ������ѡ��)��
# SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)

# ��ά��
# 1�����ɷַ�������PCA����
# print(PCA(n_components=2).fit_transform(iris.data))
# 2�������б��������LDA����
# print(LDA(n_components=2).fit_transform(iris.data, iris.target))
# 3������ֵ�ֽ⣨SVD����

#�����ھ�Ĺؼ���
# 1�����д���
#     a.���岢�д���
#     b.���ֲ��д���
# 2����ˮ�ߴ���
# 3���Զ������Σ�
# 4���־û���

# ���岢�д���
# �½�����������������ж�������ת���Ķ���
# step2_1=('TopLog', FunctionTransformer(log1p))
# �½�����������������ж�ֵ����Ķ���
# step2_2=('ToBinary', Binarizer())
# transformer_list��Ϊ��Ҫ���д���Ķ����б�
# step2=('FeatureUnion', FeatureUnion(transformer_list=[step2_1, step2_2]))

# ���ֲ��д���
# ����ȡ��������Ϊ����ѡ��Ĳ��д���
# �½�����������������ж�������ת���Ķ���
# step2_1=('TopLog', FunctionTransformer(log1p))
# �½�����������������ж�ֵ����Ķ���
# step2_2=('ToBinary', Binarizer())
# FeatureUnionExt��Ҫ�ع��ġ�
# step2=('FeatureUnionExt', FeatureUnionExt(tranfromer_list=[step2_1, step2_2, step2_3], idx_list=[[0],[1,2,3],[4]]))

# ��ˮ�ߴ�������һ�����̵��������һ�����̵����룩��
#�½�����ȱʧֵ�Ķ���
# step1 = ('Imputer', Imputer())
#�½�����������������ж�����������Ķ���
# step2_1 = ('OneHotEncoder', OneHotEncoder(sparse=False))
#�½�����������������ж�������ת���Ķ���
# step2_2 = ('ToLog', FunctionTransformer(log1p))
#�½�����������������ж�ֵ����Ķ���
# step2_3 = ('ToBinary', Binarizer())
#�½����ֲ��д�����󣬷���ֵΪÿ�����й���������ĺϲ�
# step2 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=[step2_1, step2_2, step2_3], idx_list=[[0], [1, 2, 3], [4]]))
#�½������ٻ�����
# step3 = ('MinMaxScaler', MinMaxScaler())
#�½�����У��ѡ�������Ķ���
# step4 = ('SelectKBest', SelectKBest(chi2, k=3))
#�½�PCA��ά�Ķ���
# step5 = ('PCA', PCA(n_components=2))
#�½��߼��ع�Ķ�����Ϊ��ѵ����ģ����Ϊ��ˮ�ߵ����һ��
# step6 = ('LogisticRegression', LogisticRegression(penalty='l2'))
#�½���ˮ�ߴ������
#����stepsΪ��Ҫ��ˮ�ߴ���Ķ����б����б�Ϊ��Ԫ���б���һԪΪ��������ƣ��ڶ�ԪΪ����
# pipeline = Pipeline(steps=[step1, step2, step3, step4, step5, step6])

# �Զ�������(�½������������󣬵�һ����Ϊ��ѵ����ģ�ͣ�param_gridΪ����������ɵ�����)��
# grid_search=GridSearchCV(Pipeline, param_grid={'FeatureUnionExt_ToBinary_threshold':[1.0, 2.0, 3.0, 4.0], 'LogisticRegression_C':[0.1, 0.2, 0.4, 0.8]})
# ѵ������
# grid_search.fit(iris.data, iris.target)

# �־û���
# from externals.joblib import dump, load
# ��һ������Ϊ�ڴ��еĶ��󣬵ڶ�������Ϊ��������ƣ�����������Ϊѹ������
# dump(grid_search, 'grid_search.dmp', compress=3)
# ���ļ�ϵͳ�м������ݵ��ڴ档
# grid_search=load('grid_search.dmp')





















































