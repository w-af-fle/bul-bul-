from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

class TextModel(nn.Module):
    #                   размер словаря   выходная размерность, чем больше тем больше размер итоговой матрицы
    def __init__(self, vocabulary_size, embedding_size, hidden_size, lstm_layers, lstm_dropout):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)#писываем слои модели
#embedding - реобразует слова в векторы
        self.lstm = nn.LSTM(embedding_size, hidden_size, lstm_layers, dropout=lstm_dropout, batch_first=True)
        self.dropout = nn.Dropout(0.3)#исключение переобучения
        self.fc = nn.Linear(hidden_size, 1) #положителдьный или отрицательный
        self.sigmoid = nn.Sigmoid()#тк бинарная классификация

    def forward(self, inp):
        out = inp.long()
        out = self.embedding(out)
        out = self.lstm(out)[0]#берем все данные с последнего временного шага
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

data = pd.read_csv('reviews_processed.csv')
reviews = data.processed.values
all_word = ' '.join(reviews).split()#формируем одну строку из слов через запятую, обращаясь к столбцу значения и преобраз в масслив
# print(all_word)
model_path = './lern.path'

#щздание словаря
counter = Counter(all_word)#подсчитываем количсество повторений каждого элемента
vocabulary = sorted(counter, key=counter.get, reverse=True)#сортируем по убивынию количества повторений в массиве
# print(len(vocabulary))

#реобразования слова в число
int2word = dict(enumerate(vocabulary, 1))#создаем пару члово-число начиная 1
int2word[0] = 'PAD'
word2int = {word: id for id, word in int2word.items()} #меняем слово и число меставии

#кодирование отзыво
reviews_enc = [[word2int[word] for word in review.split()] for review in reviews]
# print(reviews_enc)

#паддинг
sequence_length = 256#количество чисел в отзыве
reviews_padding = np.full((len(reviews_enc), sequence_length), word2int['PAD'], dtype=int)

#заполняем матрицу из массива с обрезанием по количеству слов
for i, row in enumerate(reviews_enc):
    reviews_padding[i, :len(row)] = np.array(row)[:sequence_length]

labels = data.label.to_numpy() #метки отзывово

train_len = 0.6
test_len = 0.5
train_last_index = int(len(reviews_padding) * train_len)

#собираем тренировочный массив матрицы отзывов              остаток от полученного индекса
train_x, remainder_x = reviews_padding[:train_last_index], reviews_padding[train_last_index:]
#массивы ожидаемых результатов
train_y, remainder_y = labels[:train_last_index], labels[train_last_index:]

test_last_index = int(len(remainder_x)*test_len)
test_x = remainder_x[:test_last_index]
test_y = remainder_y[test_last_index:]

chek_x = remainder_x[test_last_index:]
chek_y = remainder_y[test_last_index:]

#преобразуем массивы в тензоры
train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
text_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
chek_dataset = TensorDataset(torch.from_numpy(chek_x), torch.from_numpy(chek_y))

#отовим для пакетного градиентного спуска при обучении
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)
chek_loader = DataLoader(chek_dataset, batch_size=batch_size, shuffle=True)

model = TextModel(len(int2word), 256, 128, 4, 0.25)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr = 0.001)
num_epoch = 5

#функция точности предсказания
def get_accuracy(out, target):
    predicted = torch.Tensor([1 if i else 0 for i in out > 0.5])
    equals = predicted == target #вектор где сохранены 1(вернр) и 0(невернр)
    return torch.mean(equals.type(torch.FloatTensor)).item()

#по минимальной потере
test_lost_min = torch.inf
clip_grad = 1.0

for epoch in range(num_epoch):
    model.train()
    train_accuracy = 0
    train_loss = 0

    for i, (current_rewiews, target) in enumerate(train_loader):
        print('Trained (epoch %d): %d out of %d' % (epoch + 1, i, len(train_loader)))
        optimizer.zero_grad()
        out = model(current_rewiews)
        train_accuracy += get_accuracy(out, target)
        loss = criterion(out.squeeze(), target.float())
        train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    model.eval()
    test_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for i, (current_rewiews, target) in enumerate(test_loader):
            print('Test (epoch %d) %d out of %d' % (epoch + 1, i, len(test_loader)))
            out = model(current_rewiews)
            test_accuracy += get_accuracy(out, target)
            loss = criterion(out.squeeze(), target.float())
            test_loss += loss.item()

    print('Validation accuracy^ %f%%' % (test_accuracy * 100 / len(test_loader)))
    print('Validation loss: %f' % (test_loss / len(test_loader)))

    test_loss = test_loss / len(test_loader)
    if test_loss < test_lost_min:
        test_lost_min = test_loss
        torch.save(model.state_dict(), model_path)

    print('Train accuracy: %f%%' % (train_accuracy * 100 / len(train_loader)))
    print('Train loss %f' % (train_loss / len(train_loader)))

#1 epoch
#Validation accuracy^ 50.276899%
#Validation loss: 0.693464
#Train accuracy: 50.216090%
#Train loss 0.693591

#2 epoch
# Validation accuracy^ 49.960443%
# Validation loss: 0.699471
# Train accuracy: 51.747562%
# Train loss 0.686554

# 3 epoch
# Validation accuracy^ 50.385680%
# Validation loss: 0.755358
# Train accuracy: 54.281915%
# Train loss 0.655960

#Validation accuracy^ 49.940665%
# Validation loss: 0.852034
# Train accuracy: 67.299424%
# Train loss 0.595838

# Validation accuracy^ 49.940665%
# Validation loss: 0.852034
# Train accuracy: 67.299424%
# Train loss 0.595838

# Скорость обучения:
#Модель с 4 слоями LSTM будет обучаться медленнее (примерно в 1.5-2 раза)
# Точность: небольшое улучшение точности (на 1-3%)
