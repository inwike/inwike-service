import torch
import random
import numpy as np

# for i in range(0, 6):
#     cycles = random.randint(4, 8)
#     for c in range(0, cycles):
#         s = ''
#         for j in range(0, 10):
#             v = random.randint(0, 100)
#             s = s + str(v)
#             if j < 9: s += ','
#
#         print('[{},[{}]],'.format(i, s))

# Инициализация данных

competences = [
    [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
    [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
    [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
    [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
    [1, [12, 92, 35, 24, 28, 3, 36, 81, 89, 8]],
    [1, [12, 92, 35, 24, 28, 3, 36, 81, 89, 8]],
    [1, [12, 92, 35, 24, 28, 3, 36, 81, 89, 8]],
    [1, [12, 92, 35, 24, 28, 3, 36, 81, 89, 8]],
    [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
    [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
    [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
    [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
    [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
    [2, [96, 26, 76, 16, 5, 83, 0, 61, 79, 29]],
    [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
    [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
    [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
    [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
    [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
    [3, [54, 86, 92, 98, 71, 48, 63, 94, 99, 88]],
    [4, [75, 24, 35, 93, 22, 52, 52, 29, 9, 83]],
    [4, [75, 24, 35, 93, 22, 52, 52, 29, 9, 83]],
    [4, [75, 24, 35, 93, 22, 52, 52, 29, 9, 83]],
    [4, [75, 24, 35, 93, 22, 52, 52, 29, 9, 83]],
    [5, [4, 62, 88, 96, 96, 83, 45, 10, 47, 15]],
    [5, [4, 62, 88, 96, 96, 83, 45, 10, 47, 15]],
    [5, [4, 62, 88, 96, 96, 83, 45, 10, 47, 15]],
    [5, [4, 62, 88, 96, 96, 83, 45, 10, 47, 15]],
    [0, [14, 18, 81, 47, 27, 44, 54, 26, 88, 31]],
]
positions = ['Бухгалтер', 'Программист', 'Приборист', 'Инженер по Охране труда', 'Специалист по ПБ',
             'Руководитель проекта']

# Подготовка Набора тренировочных и валидационных данных

# Валидационный набор пока будет последня строка с компетенциями для каждой профессии
e_prev = -1
y_val = []
x_val = []
for e in competences:
    if e[0] != e_prev:
        e_prev = e[0]
        y_val.append(e[0])
        x_val.append(e[1])

y_max = max(y_val)
print(y_max)

# y_train = torch.tensor([e[0] for e in competences]).float()
y_train = torch.zeros(len(competences), y_max + 1).float()
for n, v in enumerate([e[0] for e in competences]):
    y_train[n, v] = 1.0

x_train = torch.tensor([e[1] for e in competences]).float()
x_train = x_train / 100
# print(x_train)

y_validation = torch.zeros(len(y_val), y_max + 1).float()
for n, v in enumerate(y_val):
    y_validation[n, v] = 1.0
# y_validation = torch.tensor(y_validation).float()
x_validation = torch.tensor(x_val).float() / 100
# print(x_validation)

# x_train.unsqueeze_(1)
# y_train.unsqueeze_(1)
# x_validation.unsqueeze_(1)
# y_validation.unsqueeze_(1)

print(len(y_train))
print(y_train)
print(y_validation)

#Конструируем Сеть
class ScopeNet(torch.nn.Module):
    def __init__(self, n_scopes, n_profession):
        super(ScopeNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_scopes, n_scopes)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_scopes, n_profession)
        #         self.act2 =  torch.nn.ReLU()
        #         self.fc3 = torch.nn.Linear(n_profession, n_profession)
        self.act3 = torch.nn.Softmax()

    #         self.fc4 = torch.nn.Linear(n_profession, n_profession)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        #         x = self.act2(x)
        #         x = self.fc3(x)
        x = self.act3(x)
        #         x = self.fc4(x)

        return x


scope_net = ScopeNet(10, 6)
print(scope_net)

# Оптимизатор
optimizer = torch.optim.Adam(scope_net.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(scope_net.parameters(), lr=0.001, momentum=0.9)

#Loss function
def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


# loss = torch.nn.CrossEntropyLoss()
# loss = torch.nn.NLLLoss()

# Training

for epoch_index in range(100):
    print('Epoch #{}'.format(epoch_index))
    running_loss = 0.0
    for i in range(len(x_train)):
        inputs = x_train[i]
        labels = y_train[i]

        optimizer.zero_grad()
        outputs = scope_net(inputs)
        loss_val = loss(outputs, labels)
        #         print(outputs)
        #         print(labels)
        loss_val.backward()
        optimizer.step()
        # печатаем статистику
        running_loss += loss_val.item()
    print('[%d] loss: %.3f' % (epoch_index + 1, running_loss))

#Validation
def predict(net, x, y):
    # y_pred = net.forward(x)
    # print(y_pred)
    outputs = net(x)
    print(outputs)

predict(scope_net, x_validation[0], y_validation[0])
predict(scope_net, x_validation, y_validation)