import numpy as np
import pickle

# 아마 당뇨병 데이터
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Load the diabetes dataset
diabetes = datasets.load_diabetes()

x_data = diabetes.data
t_data = diabetes.target

t_d = x_data[0:int(442*0.85)]
d_d = x_data[int(442*0.85):int(442*0.9)]
test_d = x_data[int(442*0.9):]

t_t = t_data[0:int(442*0.85)]
d_t = t_data[int(442*0.85):int(442*0.9)]
test_t = t_data[int(442*0.9):]

print(t_t.shape)

#수치 미분
def numerical_derivative(f,x):
  delta_x = 1e-4
  grad = np.zeros_like(x)

  it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])

  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = float(tmp_val)+delta_x
    fx1 = f(x)

    x[idx] = tmp_val - delta_x
    fx2 = f(x)

    grad[idx] = (fx1-fx2)/(2*delta_x)

    x[idx] = tmp_val
    it.iternext()

    return grad



def softmax(x):
  exp_x = np.exp(x)
  sum_exp_x = np.sum(exp_x)
  y = exp_x / sum_exp_x

  return y

# 조기종료
class EarlyStopping:
    def __init__(self, patience=5):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience
        self.w = []
        
    def step(self, loss, w):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
            self.w = w
        else:
            self.patience += 1
    
    def is_stop(self):
        return self.patience >= self.patience_limit

    def returnweight(self):
        return self.w
        
class NeuralNetwork:
    #input_nodes 특징의 개수
    def __init__(self, input_nodes):
        self.input_nodes = input_nodes
       
        self.learning_rate = 0.01
        self.w = np.random.randn(self.input_nodes)/ np.sqrt(self.input_nodes/2)
        self.b = 0
        self.z = np.zeros([1,input_nodes])
        self.a = np.zeros([1,input_nodes])
        self.target_data = None
        self.p_epoch = 0

    def change_learning_rate(self):
      self.learning_rate = float(input("학습률을 입력해주세요 : "))

    def loss_val(self,x,y):    
      y_pred=self.feed_forward(x)
      return (y_pred-y)**2

    def feed_forward(self, x):
        
        y_hat = np.sum(x*self.w)+self.b
        return y_hat

    def gradient(self, x, y):
        y_hat = self.feed_forward(x)
        w_grad = 2*x*(y_hat-y)
        b_grad = 2*(y_hat-y)

        return w_grad, b_grad


    def train(self, input_data, target_data):  
        self.target_data = target_data
        y_hat= self.feed_forward(input_data)
        w_grad = np.zeros(input_data.shape[1])
        b_grad = 0
        loss = 0
        for x,y in zip(input_data, target_data):
          loss += self.loss_val(x,y)
          w_i, b_i = self.gradient(x,y)

          w_grad += w_i
          b_grad += b_i

        self.w -= self.learning_rate*(w_grad/len(target_data))
        self.b -= self.learning_rate*(b_grad/len(target_data))
        

    def print_data(self, input, target, test_data, test_target):
      loss = 0
      for x,y in zip(input, target):
        loss += self.loss_val(x,y)
      print("epoch = ", self.p_epoch ,",  loss_val = ",loss/(input.shape[1]),"accuracy = ", str(self.accuracy(test_data,test_target))+"%")
      self.p_epoch+=1
      return loss/(input.shape[1])

    def inputweight(self, w):
      self.w = w 
           
    def accuracy(self, test_data, test_target):
        matched_list = []
        not_matched_list = []
    
        for index in range(len(test_data)):
            label = int(test_target[index])
            
            data = test_data[index]
            
            self.x = data
            predicted_num = int(self.feed_forward(data))

            if (label >= predicted_num-1) and (label <=predicted_num+1):
                matched_list.append(index)
            else:
                not_matched_list.append(index)
 
        return 100*(len(matched_list)/(len(test_data)))
        

batch_size = 128
num_epoch = 128
hidden_layer = 1
hidden_layer_size = [100]
learning_rate = 1e-4
option = input("기본 옵션을 변경하실건가요? Y 아닐 경우 아무키 : ")
if option == "Y":
  num_epoch = int(input("epoch의 수를 입력해주세요 : "))
  batch_size = int(input("배치의 크기를 입력해주세요 :"))
  hidden_layer_size.clear()
  
nn = NeuralNetwork(10)
early_stop = EarlyStopping(patience = 30)
epochs = num_epoch
nn.change_learning_rate()
training_size = t_d.shape[0]
for i in range(epochs):
  mini_batch = np.random.choice(training_size, batch_size)
  batch_training_data = t_d[mini_batch]
  batch_training_data_t = t_t[mini_batch]

  
  nn.train(batch_training_data,batch_training_data_t)
 
  loss = nn.print_data(d_d,d_t,test_d,test_t)
  early_stop.step(loss, nn.w)
  if early_stop.is_stop():
    w = early_stop.returnweight()
    nn.inputweight(w)

    print("epoch finished", i)
    break