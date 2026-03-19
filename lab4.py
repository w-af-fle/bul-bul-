# #описание модели посредствоем сверточных слоев(общие признаки)
#
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.optim import Adam
# from torchvision.datasets import CIFAR10
# import torchvision.transforms as transforms
# import torchvision.utils
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
#
# #писание модели со сверточного слоя
# class Image(nn.Module):
#     def __init__(self):
#         super(Image, self).__init__()
#         #                      вход. кол-во каналов; выход. каналы; размер ядра; шаг; поля(заполнение) дополнение пустых значений
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
#
#         #ледующий слой нормализации(среднеквадратич. отклолнения, средние значения нормализуют)
#         self.bn1 = nn.BatchNorm2d(12) #по кол-ву выходных каналов, 2D пространство
#
#         self.pool = nn.MaxPool2d(2, 2)  # уменьшение размерности данных, выбор максимального значения
#
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(24)
#
#
#         self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(24)
#
#         self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(24)
#
#         #4 - по выходному. итоговая размерность матрицы линейного слоя
#         self.fc = nn.Linear(24 * 10 * 10, 10)
#
#     def forward(self, inp):
#         out = F.relu(self.bn1(self.conv1(inp)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.pool(out)
#         out = F.relu(self.bn3(self.conv3(out)))
#         out = F.relu(self.bn4(self.conv4(out)))
#         out = out.view(-1, 24*10*10) #ункция определяет необход. размерность, второе число то, что должно получиться
#         out = self.fc(out)
#         return out
#
#
# root = "./Data_10"
# batch_size = 10
#
# #объявляем трансформацию. диапазон -1 до +1. чтобы не было переполнения
# transformations = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# #определение тренировочных данных.                   в момент загрузки преобразованные значения; чтобы брались из скаченных а не скачивались каждый раз
# train_set = CIFAR10(root=root, train=True, transform=transformations, download=True)
#
# #поставщик изображение.                   загружаем модель пакетами данных; предварительная случайная сортировка.
# train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#
# #тоже самое с тестовыми данными
# test_set = CIFAR10(root=root, train=False, transform=transformations, download=True)
#
# test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#
# # for (image, labels) in train_data_loader:
#     # print(image.shape)
#     # break
#
# model = Image()
# classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
#
# #функция тестирования точности
# def test_accuracy():
#     model.eval()
#     accuracy = 0 #количество угаданных вариантов
#     total = 0 #бщее количество
#
#     for data_test in test_data_loader:
#         images, labels = data_test
#         # print(labels)
#         # break
#         output = model(images)
#         predict = torch.max(output.data, 1)[1]#макс значения по столбцам
#         accuracy += (predict == labels).sum().item()#колво правильных угад. ответов
#         total += labels.size(0)
#         # print(accuracy, total)
#
#     return 100 * accuracy / total
#
# # print(test_accuracy())
#
# loss = nn.CrossEntropyLoss()
# #                                               уменьшает веса межнейронного взаимодействия
# optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#
# best_accuracy = 0.0
# model_save_path = "./LearnModel.pth"
# num_epochs = 4
# accuracy = test_accuracy()#осле первого прохождения точность
# model.train()
#
# # for epoch in range(num_epochs):
# #     for i, (images, labels) in enumerate(train_data_loader, 0):
# #         optimizer.zero_grad()
# #         output = model(images)
# #         error = loss(output, labels)
# #         error.backward()
# #         optimizer.step()
# #     accuracy = test_accuracy()
# #
# #     if (accuracy > best_accuracy):
# #         best_accuracy = accuracy
# #         torch.save(model.state_dict(), model_save_path)
# #
# #     print('Epoch: %d; %d%%' %(epoch+1, accuracy))
#
# #функцию вывода названий на экран перед созданием модели
# def print_labels(title, labels):
#     print(title, end=' ')
#     for i in range(10):
#         print(classes[labels[i]], end=' ')
#     print()
#
#
# load_model = Image()
# load_model.load_state_dict(torch.load(model_save_path))
# images, labels = next(iter(test_data_loader))
#
# print('True labels: ', labels)
#
# output = load_model(images) #отдаем тестовые картинки
# predict = torch.max(output, 1)[1]
# print('Predicted labels: ', predict)
#
# images = torchvision.utils.make_grid(images)
#
# images = images/2 + 0.5
#
# plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
# plt.show()
#
#

#описание модели посредствоем сверточных слоев(общие признаки)

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

#писание модели со сверточного слоя(ИЗМЕНЕНИЕ НА 3 ЯДРА)
class Image(nn.Module):
    def __init__(self):
        super(Image, self).__init__()
        #                      вход. кол-во каналов; выход. каналы; размер ядра; шаг; поля(заполнение) дополнение пустых значений
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        #ледующий слой нормализации(среднеквадратич. отклолнения, средние значения нормализуют)
        self.bn1 = nn.BatchNorm2d(12) #по кол-ву выходных каналов, 2D пространство

        self.pool = nn.MaxPool2d(2, 2)  # уменьшение размерности данных, выбор максимального значения

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)

        #4 - по выходному. итоговая размерность матрицы линейного слоя
        self.fc = nn.Linear(24 * 16 * 16, 10)

    def forward(self, inp):
        out = self.pool(F.relu(self.bn1(self.conv1(inp))))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out.view(-1, 24*16*16) #ункция определяет необход. размерность, второе число то, что должно получиться
        out = self.fc(out)
        return out
#уменьшение ядер уменьшает количество параметров, скорость обучения увеличивается

root = "./Data_10"
batch_size = 20

#объявляем трансформацию. диапазон -1 до +1. чтобы не было переполнения
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#определение тренировочных данных.                   в момент загрузки преобразованные значения; чтобы брались из скаченных а не скачивались каждый раз
train_set = CIFAR10(root=root, train=True, transform=transformations, download=True)

#поставщик изображение.                   загружаем модель пакетами данных; предварительная случайная сортировка.
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

#тоже самое с тестовыми данными
test_set = CIFAR10(root=root, train=False, transform=transformations, download=True)

test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# for (image, labels) in train_data_loader:
    # print(image.shape)
    # break

model = Image()
classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


#функция тестирования точности
def test_accuracy():
    model.eval()
    accuracy = 0 #количество угаданных вариантов
    total = 0 #бщее количество

    for data_test in test_data_loader:
        images, labels = data_test
        # print(labels)
        # break
        output = model(images)
        predict = torch.max(output.data, 1)[1]#макс значения по столбцам
        accuracy += (predict == labels).sum().item()#колво правильных угад. ответов
        total += labels.size(0)
        # print(accuracy, total)

    return 100 * accuracy / total

# print(test_accuracy())

loss = nn.CrossEntropyLoss()
#                                               уменьшает веса межнейронного взаимодействия
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

best_accuracy = 0.0
# model_save_path = "./LearnModel3.pth" #на 3 ядрах снизмлась точность, но учеличилась скорость
model_save_path = "./LearnModel3x3.pth"
num_epochs = 3
accuracy = test_accuracy()#осле первого прохождения точность
model.train()
#
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_data_loader, 0):
#         optimizer.zero_grad()
#         output = model(images)
#         error = loss(output, labels)
#         error.backward()
#         optimizer.step()
#     accuracy = test_accuracy()
#
#     if (accuracy > best_accuracy):
#         best_accuracy = accuracy
#         torch.save(model.state_dict(), model_save_path)
#
#     print('Epoch: %d; %d%%' %(epoch+1, accuracy))

#функцию вывода названий на экран перед созданием модели
def print_labels(title, labels):
    print(title, end=' ')
    for i in range(20):
        print(classes[labels[i]], end=' ')
    print()


load_model = Image()
load_model.load_state_dict(torch.load(model_save_path))
images, labels = next(iter(test_data_loader))

print('True labels: ', labels)

output = load_model(images) #отдаем тестовые картинки
predict = torch.max(output, 1)[1]
print('Predicted labels: ', predict)

count = 0
for i in range(20):
    if predict[i] == labels[i]:
        count += 1

print(f"\nПравильных ответов: {count} из 20")
print(f"Точность: {100 * count / 20}%")

images = torchvision.utils.make_grid(images)

images = images/2 + 0.5

plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
plt.show()


# Точность модели с ядрами 3x3(64% - 67% - 70%) снизилась по сравнению с моделью с ядрами 5x5(64 - 69 - 72). Скорость увеличилась
# Точность с 2 сверточными слоями стала меньше (62% - 64%). скорость увеличилась
