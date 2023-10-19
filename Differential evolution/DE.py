import numpy as np
import random
import math

#### 初始化 ####
NP = 100 # 种群个体数
iterate = 100000  #进化代数
x_l = -3 
x_r = 3 
CR = 0.1 # 交叉概率 

# objective function
def objective_value(x):
    return 2*(x**2)+3*x-1
# 初始化
G = np.linspace(-3,3,100) + random.normalvariate(0,1)

def Differential_Evolution(G):
    # 变异(随机选择一个当前个体与其他三个个体进行变异)
    index = random.sample(range(0,NP),4) # 随机选择的变异个体
    # 2*random.random() 为 F 缩放因子 
    # 交叉
    if random.random() <= CR :# j == random.randint(0,D-1)
        v_new =  G[index[1]] + 2*random.random()*(G[index[2]]-G[index[3]]) #变异
        if v_new <= x_l :
            # 任选其一
            v_new = random.random()*(x_r - x_l) + x_l  # 上下限之间取随机数来代替不在范围内的值
            v_new = x_l # 边界值吸收法
        if v_new >= x_r:
            v_new = random.random()*(x_r - x_l) + x_l
            v_new = x_r
    else:
        v_new = G[index[0]]
    
    # 适应度函数
    if objective_value(v_new) < objective_value(G[index[0]]):
        G[index[0]] = v_new 
        
            
for i in range(iterate):
    Differential_Evolution(G)
    
print("final iterating is ",G)
