from MLP import *
import numpy as np
import random
batch_size = random.randrange(10,500)
num_epoch = 100
hidden_layer = 1
hidden_layer_size = [100]
learning_rate = 1e-4
option = input("기본 옵션을 변경하실건가요? Y 아닐 경우 아무키 : ")
if option == "Y":
  hidden_layer = int(input("은닉층의 개수를 입력해주세요 : "))
  num_epoch = int(input("epoch의 수를 입력해주세요 : "))
  batch_size = int(input("배치의 크기를 입력해주세요 :"))
  hidden_layer_size.clear()
  for i in range(hidden_layer):
    print(str(i)+"번째 ", end= "")
    hl = int(input('은닉층의 차원을 입력하세요 : '))
    hidden_layer_size.append(hl)
    
    
# 훈련 및 테스트 데이터 불러오기    
training_data = np.loadtxt('dataset\mnist_train.csv', delimiter=',', dtype=np.float32)

test_data = np.loadtxt('dataset\mnist_test.csv', delimiter=',', dtype=np.float32)

input_nodes = 784
hidden_nodes = hidden_layer_size
output_nodes = 10
epochs = num_epoch
training_size = training_data.shape[0]
nn = NeuralNetwork(input_nodes, hidden_layer, hidden_nodes, output_nodes)


early_stop = EarlyStopping(patience = 15)
nn.change_learning_rate()
for i in range(epochs):
  
  
  mini_batch = np.random.choice(training_size, batch_size)
  batch_training_data = training_data[mini_batch]
  for step in range(len(mini_batch)):  # train
    
    # input_data, target_data normalize        
    target_data = np.zeros(output_nodes) + 0.01    
    target_data[int(batch_training_data[step, 0])] = 0.99
    
    input_data = ((batch_training_data[step, 1:] / 255.0) * 0.99) + 0.01
 
    nn.train(np.array(input_data, ndmin=2), np.array(target_data, ndmin=2) )
    
      
  print("epoch = ", i+1, "mini_batch_size = ", batch_size,  ",  loss_val = ", nn.loss_val(),"accuracy = ", str(nn.accuracy(batch_training_data))+"%")

  early_stop.step(nn.loss_val())
  if early_stop.is_stop():
    print("epoch finished", i)
    break

nn.save()
nn.accuracy(test_data)
# load확인용
nm = NeuralNetwork(input_nodes, hidden_layer, hidden_nodes, output_nodes)
nm.load()
nm.accuracy(test_data)
