import torch
import numpy as np
import torch.nn as nn



# # empty_tensor = torch.empty(3, 3)
# # print(empty_tensor)
# # #Создание одномерного тензора из списка
# # tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
# # print(tensor_from_list)
# #
# # #Создание одномерного тензора из списка в виде матрицы
# # tensor_from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
# # print(tensor_from_list)
# #
# # #создание тензора из массива Numpy
# # numpy_array = np.array([6, 7, 8, 9, 10])
# # tensor_from_numpy = torch.tensor(numpy_array)
# # print(tensor_from_numpy)
# #
# # #Создание тензора с единицами или нулями
# # one_tensor = torch.ones(2, 2)
# # print(one_tensor)
# # zeros_tensor = torch.zeros(2, 2)
# # print(zeros_tensor)
#
# class SimpleModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size): #конструктор с 3 параметрами, описыв-х кол-во нейронов в слоях
#         super(SimpleModel, self).__init__() #вызываем конструктор базового класса
#         self.fc1 = nn.Linear(input_size, hidden_size)#первый линейный слой и сохраняем в переменную класса
#         self.relu = nn.ReLU() #ропуск выхода функции скрытого слоя
#         self.fc2 = nn.Linear(hidden_size, output_size)#второй линейный слой и сохраняем в переменную класса
#         self.softmax = nn.Softmax(dim=1) #пропустим выход. выбираем ось столбцов, указывая размерность 1
#
#     #Функция для перехода по слоям
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.softmax(out)
#         return out
#
# input_size = 3
# hidden_size = 4
# output_size = 3
# learning_rate = 0.01
# num_epochs = 100
#
# #модель с использованием класса
# model = SimpleModel(input_size, hidden_size, output_size)
#
# # #Функция потерь,используя функцию перекрестной энтропии
# # loss = nn.CrossEntropyLoss()
# # #в качистве оптимайзера - стахостический градиентный спуск
# # optimazer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# x = torch.randn(100, input_size)
# # print(x)
#
# y = torch.randint(0, output_size, (100,))
# # print(y)
# #
# # for epoch in range(num_epochs):
# #     optimazer.zero_grad() #обнуляем оптимайзер в начале каждой эпохи
# #     out =  model(x) #прогноз прямого распространения
# #     error = loss(out, y) #потери
# #     error.backward() #обратное распространение
# #     optimazer.step() #стахостический градиентный спуск и пересчитываемвесовые коэф
# #     if (epoch + 1) % 10 == 0:
# #         print(f'Эпоха [{epoch + 1}], Потери: {error.item():.4f}')
#
# torch.save(model.state_dict(), 'SimpleModel.pth')
# model.load_state_dict(torch.load('SimpleModel.pth'))
#
# print(model(torch.randn(10, input_size)))

#первое
tensor1 = torch.randn(3, 3)
tensor2 = torch.randn(3, 3)
sum = tensor1 + tensor2
# print(sum)

umn = tensor1 * tensor2
# print(umn)

transposed = tensor2.T
# print(transposed)

meant1 = tensor1.mean()
meant2 = tensor2.mean()
# print(meant1)
# print(meant2)

max1 = tensor1.max()
max2 = tensor2.max()
# print(max1)
# print(max2)


#Используя фреймворк PyTorch, создайте нейросеть, которая будет перемножать входные 2 нейрона.
class PointNN(nn.Module):
    def __init__(self):
        super(PointNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = PointNN()
# print(model_point)

criterion = nn.MSELoss()
#в качистве оптимайзера - стахостический градиентный спуск
optimazer = torch.optim.SGD(model.parameters(), lr=0.001)

torch.manual_seed(42)

n = 100

x = torch.rand(n, 2) * 10
print(x)
y = (x[:, 0] * x[:, 1]).view(-1, 1)
print(y)

# for epoch in range(500):
#     optimazer.zero_grad()
#     out = model(x)
#     loss = criterion(out, y)
#     loss.backward()
#     optimazer.step()
#
#     if (epoch + 1) % 5 == 0:
#         print(f'Эпоха [{epoch + 1}], Потери: {loss.item():.4f}')

x_test = torch.rand(10, 2) * 10
y_test = (x_test[:, 0] * x_test[:, 1]).view(-1, 1)

predict = model(x_test)
print(predict[0])

for i in range(10):
    print(f"{x_test[i][0]:6.2f} * {x_test[i][1]:6.2f} = {y_test[i][0]:6.2f} | "
              f"Предсказание: {predict[i][0]:6.2f} | "
              f"Ошибка: {abs(predict[i][0] - y_test[i][0]):.2f}")

torch.save(model.state_dict(), 'PointNN.pth')
model.load_state_dict(torch.load('PointNN.pth', weights_only=True))

print(model(torch.rand(10, 2)))