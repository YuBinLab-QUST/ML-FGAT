# -*- coding: utf-8 -*-
from numpy import *
import csv
from matplotlib import pyplot as plt
import math
 
def load_data():                                                #读取csv文件
    dataSet1=[];dataSet2=[];label1=[];label2=[];dataSet=[]
    with open('dataSet.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            dataSet.append(line)
        for line in dataSet:
            lineArr=[]
            if(int(line[-1])==1):
                for i in range(56):
                    lineArr.append(float(line[i]))
                dataSet1.append(lineArr[:56])
                label1.append(int(line[-1]))
            else:
                for i in range(56):
                    lineArr.append(float(line[i]))
                dataSet2.append(lineArr[:56])
                label2.append(int(line[-1]))
    return mat(dataSet1),mat(label1).T,mat(dataSet2),mat(label2).T
 
def select_tr_te(dataSet,label):                                #随机各选取1000个样本作为训练集，余下为测试集
    dataTrain=[];labelTrain=[];dataTest=[];labelTest=[]
    dataIndex=range(5000)
    for i in range(1000):                                       #每类随机选取1000个样本作为训练集
        randIndex_Index=int(random.uniform(0,len(dataIndex)))
        randIndex=dataIndex[randIndex_Index]
        dataTrain.append(dataSet[randIndex])
        labelTrain.append(label[randIndex])
        del(dataIndex[randIndex_Index])
    for i in range(4000):                                       #余下4000个样本作为测试集
        randIndex=dataIndex[i]
        dataTest.append(dataSet[randIndex])
        labelTest.append(label[randIndex])
    return dataTrain,labelTrain,dataTest,labelTest
 
def class_mean(dataSet):                                         #特征均值,计算每类的均值，返回一个向量
    means=mean(dataSet,axis=0)
    mean_vectors = mat(means)
    return mean_vectors
 
def within_class_S(dataSet):                                     #计算类内散度
    m = shape(dataSet[1])[1]
    class_S=mat(zeros((m,m)))
    mean = class_mean(dataSet)
    for line in dataSet:
        x=line-mean
        class_S+=x.T*x
    return class_S
 
def class_SW(class_S1,class_S2):                                 #计算总离散度
    class_Sw=class_S1+class_S2
    return class_Sw
    
def lda():                                                       #训练样本，得出W与b
    data1,label1,data2,label2=load_data()
    dataTrain1,labelTrain1,dataTest1,labelTest1=select_tr_te(data1,label1)
    dataTrain2,labelTrain2,dataTest2,labelTest2=select_tr_te(data2,label2)
    class_S1=within_class_S(dataTrain1)
    class_S2=within_class_S(dataTrain2)
    S_w = class_SW(class_S1,class_S2)
    m1=class_mean(dataTrain1)
    m2=class_mean(dataTrain2)
    S_w1=linalg.inv(S_w)                                           #求矩阵的逆
    W=mat(S_w1*(m1-m2).T)
    b=-0.5*W.T*(m1+m2).T
    return W,b,m1,m2,dataTest1,labelTest1,dataTest2,labelTest2
 
def classifyVector(X,W,m1,m2):                                     #分类函数
    G=(X-0.5*(m1+m2))*W
    if G>0:return 1.0
    else:return 0.0
 
def testData(dataSet,label,W,m1,m2):
    m=len(dataSet)
    numTestVec=0.0
    errorCount=0
    for i in range(m):
        lineArr=[]
        numTestVec+=1.0
        lineArr=dataSet[i]
        if (classifyVector(lineArr,W,m1,m2)!=int(label[-1])):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    return errorRate
 
def Test():                                                        #调用testData函数，取平均值
    W,b,m1,m2,dataTest1,labelTest1,dataTest2,labelTest2= lda()
    errorRate1=testData(dataTest1,labelTest1,W,m1,m2)
    errorRate2=testData(dataTest2,labelTest2,W,m1,m2)
    errorRate=errorRate1+errorRate2
    # print "the error rate of this test is：%f" % (errorRate/2)