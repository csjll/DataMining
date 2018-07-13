#-*- coding:utf-8-*-

# adaboost:
# 1)有一组数据xi={x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13},需要对其进行分类；
# 2)对各个x值分配权重wi={w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13},最初分配为1/13;
# 3)通过初步的分类训练，形成一个弱分类器，主要是找到一个基本的分类准则，比如xi>x5时为1，否则为-1，此弱分类器记为G(x)；
# 4)根据这个分类器G(x)可以判断哪个数据分类是正确的，哪个数据分类是错误的，分类错误的数据点所对应的权重w的值之和为分类误差率em；
#        判断分类误差率是否在可控区间，如果在，则停止；如果不在，则继续迭代；
# 5)根据分类误差率em计算G(x)的权重am；
# 6)计算新的权重wi，用来进行
# 6)然后可以得到f(x)的函数，计算em查看是否满足分类的要求，然后再决定是否继续迭代；
# 
# 
# 伪代码：
# xi={x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13}
# wi={w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13}
# 
# //训练G(x)
# 
# //计算em、am：
# flag=True
# tt=(1/N)sum(exp(-yi*f(xi)))=multi(Zm)
# while flag:
#     em=0
#     am=0
#     for i in range(0, m):
#         em=em+wi*I(G(xi)!=yi)
#     if em!=0:
#         am=(1/2)log((1-em)/em)
#     
#     f(x)=f(x)+am*G(x)
#     
#     if em<tt:
#         flag=False
#         return f(x)
# 
#     zm=wi*exp(-am*yi*G(x))
#     wi=(w(i-1))/zm*exp(-am*yi*G(x))
# 
#     
    




