import numpy as np
import pickle
import logistic_regression
# 훈련 및 테스트 데이터 불러오기 
training_data = np.loadtxt('./mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('./mnist_test.csv', delimiter=',', dtype=np.float32)

batch_size = 128
num_epoch = 128
learning_rate = 1e-4
option = input("기본 옵션을 변경하실건가요? Y 아닐 경우 아무키 : ")
if option == "Y":
  num_epoch = int(input("epoch의 수를 입력해주세요 : "))
  batch_size = int(input("배치의 크기를 입력해주세요 :"))


from datetime import datetime 
input_nodes = 784
output_nodes = 10
epochs = num_epoch
training_size = training_data.shape[0]
nn = NeuralNetwork(input_nodes, output_nodes)

start_time = datetime.now()
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
print(nn.accuracy(test_data))


nm = NeuralNetwork(input_nodes, output_nodes)
nm.load()
print(nm.accuracy(test_data))
end_time = datetime.now() 
print("\nelapsed time = ", end_time - start_time) 
