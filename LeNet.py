import time
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

torch.backends.cudnn.benchmark = True

transforms = transforms.Compose([transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (1.0,))])

train_data_set = MNIST('/Users/haigangliu/ImageData/MNIST/',
                       download= True,
                       train= True,
                       transform= transforms
                      )

test_data_set = MNIST('/Users/haigangliu/ImageData/MNIST/',
                      download= True,
                      train= False,
                      transform = transforms)

train_loader = torch.utils.data.DataLoader(
                 dataset = train_data_set,
                 batch_size = 300,
                 num_workers = 6,
                 shuffle = True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_data_set,
                batch_size=300,
                num_workers = 6,
                shuffle=False)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
            )

        self.fc = nn.Sequential(
             nn.Linear(in_features=120, out_features=84),
             nn.ReLU(),
             nn.Linear(in_features=84, out_features=10),
             nn.LogSoftmax(dim = 1)
        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 120)
        output = self.fc(output)
        return output

model = LeNet5().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

num_epoch = 100
guess_counter = []; 
guess_counter_test = []


for epoch in range(num_epoch):
    since = time.time()
    correct_guesses = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)

        loss_cumsum = loss_cumsum + loss.item()
        correct_guesses = correct_guesses + torch.sum(predictions == labels)

        loss.backward()
        optimizer.step()

    guess_counter.append(correct_guesses)

    model.eval()

    correct_guesses = 0

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)

        loss_cumsum = loss_cumsum + loss.item()
        correct_guesses = correct_guesses + torch.sum(predictions == labels)

    guess_counter_test.append(correct_guesses)

    time_ = time.time() - since
    print("the {}th epoch is finished in {:.3f}".format((epoch+1), time_))

import numpy as np
import matplotlib.pyplot as plt

plt.plot(range(num_epoch), np.array(guess_counter)/60000.0, "r")
plt.plot(range(num_epoch), np.array(guess_counter_test)/10000.0, "b")
plt.show()
