#Получение данных
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root = "./Data_10"
batch_size = 10

#объявляем трансформацию. диапазон -1 до +1. чтобы не было переполнения
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

#определение тренировочных данных.                   в момент загрузки преобразованные значения; чтобы брались из скаченных а не скачивались каждый раз
train_set = datasets.CIFAR10(root=root, train=True, transform=transformations, download=True)

#поставщик изображение.                   загружаем модель пакетами данных; предварительная случайная сортировка.
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

#тоже самое с тестовыми данными
test_set = datasets.CIFAR10(root=root, train=False, transform=transformations, download=True)

test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

