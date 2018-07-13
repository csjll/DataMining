# -*- coding: utf-8 -*-
from celery.backends import rpc

# 1、mapreduce篇：
    # 1）用mapreduce如何实现好友推荐：在一堆单词中找出出现次数最多的k个；
    #     1.1--共同好友：a是b的好友，b是c的好友，则可以将a和c推荐到一起；
    #         a:h、s、b、d、s
    #         b:s、b、d、a
    #         第一步：先运算a是谁的好友；
    #         第二步：确定有a的共同好友对数；
    #         第三步：列出所有的好友对h-s：a；
    #         第四步：汇总相同的key值；
    #     1.2、在一堆单词中找出出现次数最多的k个：
    #         第一步：各个单词在不同的文件中；
    #         第二步：用不同的map获取到各个切片文件中单词的个数；
    #         第三步：使用reduce的方法：将各个map的count结果传输到reduce中，在reduce中汇总全部的word数据，排序并呈现出计算结果；
    #              不适用reduce的方法：一个是防止数据倾斜（在map中添加一个文件，用来存储各个map处理的结果，然后用文件将各个map串行在一起）；
    #                             一个是利用缓存进行排序（）；
    #     
    # 2）栈与队列的设计、设计一个栈，取出栈中最大值；
        # 栈：先进后出；
        # 队列：先进后出；
        # 栈的方法：入栈：s.push(x);出栈：s.pop();访问栈顶：s.top();判断栈空：s.empty();访问栈中的元素个数：s.size();
        # 队列的方法：入队：q.push();出队：q.pop();访问队首元素：q.front();访问队尾元素：q.back();判断队列为空：q.empty();访问队列中的元素个数：q.size();
        # 开辟一个新的存储最大值索引的栈，该栈和存储数据的栈保持同步，即栈顶指针同时作用于这两个栈；
            # 2.1、遍历存储数据的栈；
            # 2.2、取出来的两个数据进行比较，将max存放在新的栈中；
            # 2.3、遍历完成后将新栈中的值返回；
    # 3）求多叉树深度：
        # 遍历多叉树每一个层级的子节点个数，直到子节点个数为零；
        # ans=-1;
        # def dfs(node x, int deep):
        #     if(x.sons.size()==0){
        #         ans = max(ans, deep);
        #         return;
        #         }
        #     else{
        #         for(int i=0; i<x.sons.size();i++){
        #             dfs(x.sons[i], deep+1);
        #             }
        #     }
            
    
    # 4）hadoop原理；
        # 4.1、是一套并行计算架构与分布式文件系统；    
        # 4.2、介绍YARN的原理：client与ResourceManager、NodeManager两部分之间的资源协调；
            # ResourceManager：承接了JobTracker的功能并做了添加，由task队列、scheduler组成；
            # NodeManager：承接了TaskTrack的功能并做了添加，由MrAppMaster、container、task三部分组成；
            
            # YARN的原理：
            # 1)client首先提交一个application给到rm，rm接收到application后会查看可以存放文件的路径以及app_id；
            # 2)client接收到这个app_id以及路径，将对应的job.xml文件、jar文件等存放在HDFS中对应的路径下；
            # 3)存放完成后，client会向rm发送传送完成的指令，rm接收到这个指令后会形成一个task，并将这个task存放在队列中；
            # 4)nodemanager会定时向rm请求，查看task队列中是否存在task，如果存在task，则生成容器container；
            # 5)container容器在形成的时候会将运行所需要的资源：内存、CPU、jar文件（在HDFS中，dn从HDFS中取出数据文件）等全部汇总到container中；
            # 6)container生成的同时还在其中产生一个主管MrAppMaster，这个MrAppMaster会激活map中所有的切片，同时生成多个待运行的maptask；
            # 7)MrAppMaster紧接着会向rm提出申请，确定运行这些maptask以及运行所在的服务器，在这些服务器上生成对应个数的container；
            # 8)将运行这些maptask所需要的资源CPU等存放在container中；
            # 9)资源协调好后，MrAppMaster会将maptask发送到container中，并通过java -jar XXX启动所有的maptask；
            # 10)当maptask运行完成后，会在对应的服务器上保留一个结果文件，供reduce调用，同时maptask将运行结果告诉MrAppMaster；
            # 11)MrAppMaster向rm提出申请，申请reduce运行所需要的资源以及运行位置；
            # 12)在NodeManager中形成container，存放reducetask并运行，reducetask运行的时候会向maptask所在的服务器找map的运行结果文件；
            # 13)当reducetask运行完成后，会将运行结果存储在服务器上，并通知MrAppMaster，MrAppMaster会销毁这次的task运行；
        
            # ResourceManager：
            # 1）处理client请求；
            # 2）资源分配调度；
            # 3）启动/监控ApplicationMaster；
            # 4）监控NodeManager；
            
            # NodeManager：
            # 1）管理单个节点资源；
            # 2）处理AppMaster的命令；
            # 3）处理RM资源；
            
            # MrAppMaster：
            # 1）数据切分；
            # 2）申请资源、分配任务；
            # 3）任务监控、容错；
            
            # container：
            # 1）对运行环境以及相应的资源进行封装；
            # 2）每个任务一个container，不能共用；
        
        # 4.3、介绍MapReduce的原理：
            # map中的动作：a.切分；b.分组；c.分区；d.排序；e.发送给reduce；
            # 在map中数据被一行行的读取，然后进行切分；
            # HDFS——》map——》shuffle——》reduce——》HDFS;
            
            
            
        # 4.4、介绍HDFS的原理：client与namenode、datanode两个节点之间的数据传输；
            # HDFS的原理：
            # 1）client会向namenode请求写数据，nn返回可以存储，并返回相应的目录信息；
            # 2）client接收到存储目录，进行切分处理之后告诉nn，要进行存储第一个block（存储一次向nn请求一次）；
            # 3）返回block需要的服务器名称；
            # 4）client连接对应的服务器，在需要存放block的服务器上建立pipeline；
            # 5）pipeline连接完成后，client会将对应的数据存储到datanode服务器上；
            # 6）存储完成后client会将结果返回给nn，同时datanode会将最新的目录信息传给nn，以更新nn目录；
            
            # namenode的运行状态：
            # 1）namenode中主要有内存、edits、fsimage三部分组成：client中的数据进入到nn中后会先进入到edits中，将对应的操作存入到edits中，然后将fsimage中的数据逐条在内存中运行；
            # 2）一定时间之后，snn会向nn发起同步请求，nn接收到这一请求后切换现在在运行的日志，然后将edits、fsimage两部分打包下载到snn中，snn会将这一数据放到内存中；
            # 3）edits与fsimage会在内存中将数据合并，合并后的数据备份在snn中的同时上传到nn中一份，以此实现namenode数据同步；
            # 4）在HA中，为方便实现高可用，可以讲edits单独放到一个服务器集群中，然后用zookeeper检测edits集群的运行状况；
            
            # datanode的运行状态：
            # 1）datanode的功能主要有两个：一个是心跳，即每隔一定时间链接一次namenode，保证自身的存活；
            # 2）以block的形式存储业务数据，实现多组数据分布式存储；
 
    # 6）map如何切割数据；
        # 通过使用fileInputFormat来切分，切分的方式主要有两个：
        # 一是按照文件来分：一个文件一个切片；
        # 二是按照数据大小来分：128M超过数据量则切分到下一个切片中，用到的方法为split();
    # 7）如何处理数据倾斜；

    # 8）join的mr代码如何写；
    
    # 9)rpc框架：
    
    # 10）多线程：
    
    # 11）spring架构：

    













# 2、机器学习：
    # 1）如何在海量数据中查找给定部分数据最相似的top200向量；
    # 2）怎么衡量两个商品的性价比；
    # 3）动态规划；
    # 4）各个损失函数的区别，使用场景，如何并行化（并行算法的几种优化方法），有哪些关键参数；
    # 5）LR、FFM、SVM、RF、KNN、EM、Adaboost、PageRank、GBDT、Xgboost、HMM、DNN、CNN、RNN、LSTM 
    # 6）推荐算法（基于协同过滤的推荐、基于统计学的推荐、基于内容的推荐）
    # 7）聚类算法（各种聚类类型）
    # 8）图像、自然语言
    # 9）XGB和GBDT的评估函数：F值 mae logloss AUC MAP@N
    # 10）优化方法：随机梯度下降、牛顿拟牛顿原理、生成模型、判别模式、线性分类和非线性分类都有哪些模型；
    # 11）SVM核技巧原理、如何选择核函数、特征选择方法有哪些、
    # 12）常见融合框架原理、优缺点、bagging、boosting、为什么融合能提升效果、
    # 13）信息熵和基尼指数的关系（信息熵在x=1处一阶泰勒展开就是基尼指数）；
    # 14）如何克服欠拟合和过拟合、L0、L1、L2正则化（能画等高线）
    # 15）模型性能评估=方差+偏差+噪声
    
    
    # 16）SVM、LR、决策树的对比：
    # 17）GBDT与决策森林的对比：
    # 18）如何判断函数是凸函数还是非凸函数：
    # 19）什么是对偶、最大间隔、软间隔？
    # 20）卷积神经网络与DBN有什么区别？
    # 21） 采用EM算法解释Kmeans算法？
    # 22）如何进行实体识别？

































