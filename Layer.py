import numpy as np

class Layer(object):
    def activiation(self,x,name):
        if(name=="Sigmoid"):
            return 1/(1+np.exp(-x))
        if(name=="None"):
            return x
        if(name=="Relu"):
            return (np.abs(x) + x) / 2
        print("name error")

    def d_activiation(self,x,name):
        if(name=="Sigmoid"):
            return self.activiation(x,name)*(1-self.activiation(x,name))
        if(name=="None"):
            return 1
        if(name=="Relu"):
            return np.where(x > 0, 1, 0)