import numpy as np 
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, zero_one_loss,hamming_loss,accuracy_score,multilabel_confusion_matrix
from tensorflow.keras import models
from sklearn.metrics import label_ranking_loss
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import  Dense
from tensorflow.keras.layers import Dropout

class MyMultiHeadAttention(Layer):
    def __init__(self,output_dim,num_head,kernel_initializer='glorot_uniform',**kwargs):
        self.output_dim=output_dim
        self.num_head=num_head
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MyMultiHeadAttention,self).__init__(**kwargs)
        
    def build(self,input_shape):                          
        self.W=self.add_weight(name='W',shape=(self.num_head,3,input_shape[2],self.output_dim),
           initializer=self.kernel_initializer,
           trainable=True)
        #  
        self.Wo=self.add_weight(name='Wo',shape=(self.num_head*self.output_dim,self.output_dim),
        # self.Wo  
           initializer=self.kernel_initializer,
           trainable=True)
        self.built = True
    def call(self,x):
        for i in range(self.W.shape[0]):# 多个头循环计算
            q=K.dot(x,self.W[i,0])
            k=K.dot(x,self.W[i,1])
            v=K.dot(x,self.W[i,2])
            #print('q_shape:'+str(q.shape))
            e=K.batch_dot(q,K.permute_dimensions(k,[0,2,1]))#把k转置，并与q点乘
            e=e/(self.output_dim**0.5)
            e=K.softmax(e)
            o=K.batch_dot(e,v)
            if i ==0:
                outputs=o
            else:
                outputs=K.concatenate([outputs,o])
        z=K.dot(outputs,self.Wo)
        return z
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)
#导入数据
data=pd.read
label=pd.read
print(data.shape)
print(label.shape)
num_class=4   
X = np.array(data)
y = np.array(label)
n=np.shape(X)[1]
label_y=[]
for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j]==1:
                label_y.append(j+1)
                break
print(label_y)
def to_class(pred_y):
    for i in range(len(pred_y)):
        pred_y[i][pred_y[i]>=0.5]=1
        pred_y[i][pred_y[i]<0.5]=0
    return pred_y

def build_model(input_dim,num_class):
    model = models.Sequential()
    model.add(MyMultiHeadAttention())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_class, activation = 'softmax',name="Dense_2"))
    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics =['accuracy'])
    
    return model

[sample_num,input_dim]=np.shape(X)
Model=build_model(input_dim,num_class)
num_iter=0
yscore=np.ones((1,num_class))*0.5
yclass=np.ones((1,num_class))*0.5
ytest=np.ones((1,num_class))*0.5

def evaluate(y_gt, y_pred, threshold_value=0.5):
    print("thresh = {:.6f}".format(threshold_value))
    y_pred_bin = y_pred >= threshold_value
    
    OAA=accuracy_score(y_gt, y_pred_bin)
    print("accuracy_score = {:.6f}".format(OAA))
    mAP = average_precision_score(y_gt, y_pred)
    print("mAP = {:.2f}%".format(mAP * 100))
    score_F1_weighted = f1_score(y_gt, y_pred_bin, average="weighted")
    print("score_F1_weighted = {:.6f}".format(score_F1_weighted))
    Mconfusion_matri=multilabel_confusion_matrix(y_gt, y_pred_bin)
    print("混淆矩阵",Mconfusion_matri)
    h_loss = hamming_loss(y_gt, y_pred_bin)
    print("Hamming_Loss = {:.6f}".format(h_loss))
    R_loss = label_ranking_loss(y_gt, y_pred_bin)
    print("ranking_loss= {:.6f}".format(R_loss))
    z_o_loss = zero_one_loss(y_gt, y_pred_bin)
    print("zero_one_loss = {:.6f}".format(z_o_loss))
    
