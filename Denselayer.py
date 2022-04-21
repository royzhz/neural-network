import numpy as np
import Layer
class Dense(Layer.Layer):
    def __init__(self,shape,act_name="Sigmoid",is_output_layer=False):#建立该层矩阵,shape为一个二元组，第一个为输入神经元数量，第二个为本层有多少神经元
        self.weight=np.random.rand(shape[0],shape[1])
        self.weight_b=np.random.rand(1,shape[1])
        self.act_name=act_name
        self.shape=shape
        self.is_output_layer=is_output_layer
        return

    def forword_propagate(self,input):#input为向量输入，返回计算后的输出矩阵
        self.input=input
        self.v=input.dot(self.weight)+self.weight_b
        return self.activiation(self.v,self.act_name)

    def backword_propagate(self,error,before_weight,learning_rate):#后面传递来的误差，返回自己的误差，并更新权值
        self.past=self.weight
        if(self.is_output_layer==False):
            theta = (before_weight.dot(error))*(self.d_activiation(self.v.T,self.act_name))
        else:
            theta=(error*(self.d_activiation(self.v,self.act_name))).T

        self.weight=self.weight+(theta.dot(self.input)).T*learning_rate
        self.weight_b=self.weight_b+theta.T*learning_rate
        return theta,self.past