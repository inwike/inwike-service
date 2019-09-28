import torch
import os
from django.conf import settings

NET_PATH = 'net.dat'

# competences = [
#     [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
#     [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
#     [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
#     [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
#     [1, [12, 92, 35, 24, 28, 3, 36, 81, 89, 8]],
#     [1, [12, 92, 35, 24, 28, 3, 36, 81, 89, 8]],
#     [1, [12, 92, 35, 24, 28, 3, 36, 81, 89, 8]],
#     [1, [12, 92, 35, 24, 28, 3, 36, 81, 89, 8]],
#     [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
#     [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
#     [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
#     [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
#     [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
#     [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
#     [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
#     [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
#     [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
#     [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
#     [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
#     [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
#     [4, [75, 24, 35, 93, 22, 52, 52, 29, 9, 83]],
#     [4, [75, 24, 35, 93, 22, 52, 52, 29, 9, 83]],
#     [4, [75, 24, 35, 93, 22, 52, 52, 29, 9, 83]],
#     [4, [75, 24, 35, 93, 22, 52, 52, 29, 9, 83]],
#     [5, [4, 62, 88, 96, 96, 83, 45, 10, 47, 15]],
#     [5, [4, 62, 88, 96, 96, 83, 45, 10, 47, 15]],
#     [5, [4, 62, 88, 96, 96, 83, 45, 10, 47, 15]],
#     [5, [4, 62, 88, 96, 96, 83, 45, 10, 47, 15]],
#     [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
# ]
# positions = ['Бухгалтер', 'Программист', 'Приборист', 'Инженер по Охране труда', 'Специалист по ПБ',
#              'Руководитель проекта']

# Валидационный набор пока будет последня строка с компетенциями для каждой профессии
# e_prev = -1
# y_val = []
# x_val = []
# for e in competences:
#     if e[0] != e_prev:
#         e_prev = e[0]
#         y_val.append(e[0])
#         x_val.append(e[1])
# y_max = max(y_val)
# print(y_max)

# Подготовка Набора тренировочных и валидационных данных
def prepare_data(x, y, prof_count, divider = 1):
    # преобразуем метки ответов в одномерные вектора
    y_train = torch.zeros(len(y), prof_count).float()
    for n, v in enumerate(y):
        y_train[n, v] = 1.0

    # Готовим тренировочный DataSet
    x_train = torch.tensor(x).float() / divider

    return (x_train, y_train)

#Конструируем Сеть
class ScopeNet(torch.nn.Module):
    def __init__(self, n_scopes, n_profession):
        super(ScopeNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_scopes, n_scopes)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_scopes, n_profession)
        self.act3 = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act3(x)
        return x

# Создание Neiural Network
# scope_net = ScopeNet(10, 6)
# print(scope_net)

# Оптимизатор
# optimizer = torch.optim.Adam(scope_net.parameters(), lr=0.01)

#Loss function
# def loss(pred, target):
#     squares = (pred - target) ** 2
#     return squares.mean()

#Train Function
def train_network(net, x, y, num_epochs, log=0):
    def loss(pred, target):
        squares = (pred - target) ** 2
        return squares.mean()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch_index in range(num_epochs):
        running_loss = 0.0
        for i in range(len(x)):
            inputs = x[i]
            labels = y[i]

            optimizer.zero_grad()
            outputs = net(inputs)
            loss_val = loss(outputs, labels)
            loss_val.backward()
            optimizer.step()
            # Вычисляем статистику
            running_loss += loss_val.item()
        if log>0:
            print('Epoch #[%d] loss: %.3f' % (epoch_index + 1, running_loss))

def save_network(net):
    torch.save(net.state_dict(), NET_PATH)

def load_network(net):
    if os.path.exists(NET_PATH):
        net.load_state_dict(torch.load(NET_PATH))
        net.eval()

#Validation
def predict(net, x, y):
    # y_pred = net.forward(x)
    outputs = net(x)
    print(outputs)

# (x_train, y_train) =  prepare_data([e[1] for e in competences], [e[0] for e in competences], y_max)
# (x_validation, y_validation) =  prepare_data(x_val, y_val, y_max)
# train_network(scope_net, x_train, y_train, 100, 1)
# save_network(scope_net)
# load_network(scope_net)
#
# predict(scope_net, x_validation[0], y_validation[0])
# predict(scope_net, x_validation, y_validation)