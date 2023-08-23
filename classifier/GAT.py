from copy import deepcopy
import torch
import pandas as pd
import dgl
import random
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, zero_one_loss,hamming_loss,accuracy_score,multilabel_confusion_matrix
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error
from torch.optim.lr_scheduler import StepLR
from numpy.random import seed
from tqdm import tqdm
from torch.backends import cudnn
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True         #固定网络结构的模型优化
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_class=4
def load_data():
    data=pd.read
    label=pd.read
    X = np.array(data)
    y = np.array(label)
    label_y=[]
    features = torch.FloatTensor(X)
    y = torch.from_numpy(np.array(y).astype(np.float64))
    labels = torch.squeeze(y)
    g = dgl.knn_graph(features, 5, algorithm='bruteforce-blas', dist='cosine')
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, features, labels, label_y
g, features, labels,label_y = load_data()
sepscores = []
[sample_num,input_dim]=np.shape(features)
out_dim=num_class
probas_cnn=[]
tprs_cnn = []
sepscore_cnn = []

def to_class(pred_y):
    for i in range(len(pred_y)):
        pred_y[i][pred_y[i]>=0.5]=1
        pred_y[i][pred_y[i]<0.5]=0
    return pred_y

def train(model,data,labels,train_index):
    optimizer = torch.optim.Adam()
    loss_function = torch.nn.BCELoss().to(device)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    min_epochs = 10
    min_val_loss = 5
    best_model = None
    model.train()
    for epoch in tqdm(range()):
        out = model(data)
        optimizer.zero_grad()
        out= out.to(torch.float32)
        labels=labels.to(torch.float32)
        loss = loss_function(out[train_index], labels[train_index])
        print(labels[train_index])
        loss.backward()
        optimizer.step()
        if loss < min_val_loss and epoch + 1 > min_epochs:
            min_val_loss = loss
            best_model = deepcopy(model)
        print('Epoch {:03d} train_loss {:.4f} '.format(epoch, loss.item(), loss))
      
    return best_model


def test(model, data,test_index):
    test_pre=[]
    model.eval()
    pred = model(data)
    test_pre=pred[test_index]
    return test_pre

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
    
