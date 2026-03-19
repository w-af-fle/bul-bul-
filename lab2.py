import torch
import numpy as np
import torch.nn as nn

# class TwoHiddenLayerModel(nn.Module):
#     def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
#         super(TwoHiddenLayerModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden1_size)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden1_size, hidden2_size)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(hidden2_size, output_size)
#         self.softmax = nn.Softmax(dim=1) #озволяет получить вероятности для классов для каждого отдельного примера
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.softmax(out)
#
#         return out
#
# input_size = 4
# hidden1_size = 8
# hidden2_size = 6
# output_size = 3
# learning_rate = 0.001
# num_epochs = 100
#
# model = TwoHiddenLayerModel(input_size, hidden1_size, hidden2_size, output_size)
# criterion = nn.CrossEntropyLoss() #функция потерь
# optimizer = torch.optim.Adam(model.parameters(), learning_rate) #более быстрое схождение
#
# np.random.seed(0)
# x = np.random.rand(100, input_size).astype(np.float32)
# # print(x)
#
# y = np.random.randint(output_size, size = 100)
#
# x = torch.from_numpy(x)
# y = torch.from_numpy(y).long()
#
# for epoch in range(num_epochs):
#     optimizer.zero_grad() #обнуляем оптимайзер на каждом шаге
#     out = model(x)
#     loss = criterion(out, y) #ошибка
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         print(f'epoch {epoch + 1}, loss: {loss.item():.4f}')
#
# x_test = np.random.rand(10, input_size).astype(np.float32)
# x_test = torch.from_numpy(x_test)
# predictions = model(x_test)
#
# # print(predictions)
#
# predicted = torch.max(predictions, 1) #метки классов
# print(predicted)

class TwoHiddenLayerModel(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
        super(TwoHiddenLayerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(hidden3_size, output_size)
        self.sigmoid4 = nn.Sigmoid() #озволяет получить вероятности для классов для каждого отдельного примера

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid2(out)
        out = self.fc3(out)
        out = self.tanh3(out)
        out = self.fc4(out)
        out = self.sigmoid4(out)

        return out

input_size = 5
hidden1_size = 8
hidden2_size = 6
hidden3_size = 8
output_size = 1
learning_rate = 0.01
num_epochs = 100

model = TwoHiddenLayerModel(input_size, hidden1_size, hidden2_size, hidden3_size, output_size)
criterion = nn.BCELoss() #функция потерь
optimizer = torch.optim.Adam(model.parameters(), learning_rate) #более быстрое схождение

np.random.seed(0)
x = np.random.rand(100, input_size).astype(np.float32)
# print(x)

y = np.random.randint(0, 2, size = 100).astype(np.float32)

x = torch.from_numpy(x)
y = torch.from_numpy(y).view(-1,1)


for epoch in range(num_epochs):
    optimizer.zero_grad() #обнуляем оптимайзер на каждом шаге
    out = model(x)
    loss = criterion(out, y) #ошибка
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}, loss: {loss.item():.4f}')

x_test = torch.rand(10, input_size)

with torch.no_grad(): # Градиенты НЕ вычисляются
    probs = model(x_test)
    predictions = (probs >= 0.5).int()


print(probs)
print(predictions)

predicted = torch.max(predictions, 1) #метки классов
# print(predicted)
