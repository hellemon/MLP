import numpy as np
import pickle
# 가중치
W = np.random.rand(1,1)
b = np.random.rand(1)

#시그모이드 함수
def sigmoid(x):
  return 1/ (1+np.exp(-x))


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


def error_val(x, t):
    delta = 1e-7    # log 무한대 발산 방지
    
    z = np.dot(x,W) + b
    y = sigmoid(z)
    # cross_entropy
    return  -np.sum( t*np.log(y + delta) + (1-t)*np.log((1 - y)+delta ) )  

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
        
    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1
    
    def is_stop(self):
        return self.patience >= self.patience_limit
        
class NeuralNetwork:
    def __init__(self, input_nodes, output_nodes):
        self.input_nodes = input_nodes
       
        self.output_nodes = output_nodes
 
        self.learning_rate = 1e-3
        self.w = []
        self.b = []
        self.z = []
        self.a = []
        
       
        self.w.append(np.random.randn(self.input_nodes, self.output_nodes)/ np.sqrt(self.input_nodes/2))
        self.b.append(np.random.rand(self.output_nodes))

        
        
        self.z.append(np.zeros([1,input_nodes]))
        self.a.append(np.zeros([1,input_nodes]))


        self.z.append(np.zeros([1,output_nodes]))
        self.a.append(np.zeros([1,output_nodes]))
        
    def change_learning_rate(self):
      self.learning_rate = float(input("학습률을 입력해주세요 : "))

    def loss_val(self):
        
        delta = 1e-7    # log 무한대 발산 방지
        
     
        self.z[0]=(self.input_data)
        self.a[0]=(self.input_data)
        
       
        
        self.z[1]=(np.dot(self.a[0],self.w[0])+self.b[0]) 
        self.a[1]=(sigmoid(self.z[1]))
   
        lambd = 0.001 #정규화 상수
        squared = 0
        for i in self.w:
          squared += np.sum((i**2))
          

        cross_entropy_front  = -np.sum( self.target_data*np.log(self.a[-1] + delta)) # 손실함수

        L2_regularization_cost_back = (lambd/2) * (squared) # 정규화
        L2_regularization_cost_front =  cross_entropy_front  # 손실함수

        
        return  L2_regularization_cost_front+L2_regularization_cost_back
    def feed_forward(self):
        
        delta = 1e-7    # log 무한대 발산 방지
        
     
        self.z[0]=(self.input_data)
        self.a[0]=(self.input_data)
        
       
        
        self.z[1]=(np.dot(self.a[0],self.w[0])+self.b[0]) 
        self.a[1]=(sigmoid(self.z[1]))
   
        lambd = 0.001 #정규화 상수
        squared = 0
        for i in self.w:
          squared += np.sum((i**2))
          

        cross_entropy_front  = -np.sum( self.target_data*np.log(self.a[-1] + delta)) # 손실함수

        L2_regularization_cost_back = (lambd/2) * (squared) # 정규화
        L2_regularization_cost_front =  cross_entropy_front  # 손실함수

        
        return  L2_regularization_cost_front+L2_regularization_cost_back
        

      
      
    def train(self, input_data, target_data):  
        
        self.target_data = target_data    
        self.input_data = input_data
        
        loss_val = self.feed_forward()
        
        # 출력층 loss를 구함
                      
      
      
        loss = (self.a[-1]-self.target_data) * self.a[-1] * (1-self.a[-1])

        self.w[-1] = self.w[-1] - self.learning_rate * np.dot(self.a[-2].T, loss)
        self.b[0] = self.b[-1] - self.learning_rate * loss
          


    def save(self):
      with open("w_save.pkl","wb") as f:
        pickle.dump(self.w, f)
      
      with open("b_save.pkl","wb") as f:
        pickle.dump(self.b, f)
      
      

      
    def load(self):
      with open("w_save.pkl","rb") as f:
        self.w = pickle.load(f)
      with open("b_save.pkl","rb") as f:
        self.b = pickle.load(f)

      
       


    def predict(self, input_data):


  
      self.z[1] = np.dot(input_data, self.w[0]) + self.b[0]
      self.a[1] = sigmoid(self.z[1])
       

      predicted_num = np.argmax(self.a[-1])
      return predicted_num
        
    def accuracy(self, test_data):
        matched_list = []
        not_matched_list = []
        
        for index in range(len(test_data)):
            label = int(test_data[index, 0])
            
            data = (test_data[index, 1:]/255.0*0.99)+ 0.01
            
            predicted_num = self.predict(data)
            
            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
                
 
        return 100*(len(matched_list)/(len(test_data)))
        
       












