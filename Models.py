import numpy as np
from Denselayer import Dense
import math
import matplotlib.pyplot as plt
class model:
    def __init__(self):
        self.layers=[]
        return
    def add(self,layer):
        self.layers.append(layer)
        return

    def train(self,X,y,learn_rate=0.01,times=3):
        cost=0
        for i in range(times):
            for j in range(len(X)):
                s=X[j]
                s.resize((1,X[j].shape[0]))

                for lay in self.layers:
                    s=lay.forword_propagate(s)

                lose=y[j]-s
                cost+=abs(lose)
                before_max=np.ones((self.layers[len(self.layers)-1].shape[1],1))
                for k in range(len(self.layers)):
                    index=len(self.layers)-k-1
                    lose,before_max=self.layers[index].backword_propagate(lose,before_max,learn_rate)


            if(i%1000==0 and i!=0):
                print(cost)
                cost=0
        return

    def predict(self,X):#X为数据集
        x=X
        for i in self.layers:
            x = i.forword_propagate(x)
        return x

# 加载数据
def load_data():
    """
    加载数据集
    """
    x = np.arange(0.0,10,1)
    y =np.sin(2*np.pi*x/10)*10
    # 数据可视化
    return x,y
#进行测试

model=model()
x,y = load_data()
x = x.reshape(10,1)
y = y.reshape(10,1)


model.add(Dense((1,4),"Relu"))
model.add(Dense((4,4),"Relu"))
model.add(Dense((4,1),"None",True))
model.train(x,y,0.001,20000)
plt.scatter(x,y)
t_x=np.arange(0,10,0.2)
t_x.resize((50,1))
plt.scatter(t_x,model.predict(t_x))
plt.show()

