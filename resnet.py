import warnings
warnings.filterwarnings('ignore')
import torch as t
import torchvision as tv
import matplotlib.pyplot as mp

# data --------------------------------------------------------------------------------------------------------------
train_val_data = tv.datasets.CIFAR10(root='data/', transform=tv.transforms.ToTensor())
test_images = tv.datasets.CIFAR10(root='data/', train=False)
test = tv.datasets.CIFAR10(root='data/', train=False, transform=tv.transforms.ToTensor())
targets = ('airplane', 'automobile', 'bird',   'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# data --------------------------------------------------------------------------------------------------------------

# data preprocessing

from torch.utils.data import DataLoader, random_split

train, val = random_split(train_val_data, [40000, 10000])
batch_size = 130                                                              # h1
train = DataLoader(train, batch_size=batch_size, shuffle=True)
val = DataLoader(val, batch_size=batch_size)
test_loader = DataLoader(test, batch_size=batch_size)

#modelling
import torch.nn as nn
from torch.nn.functional import softmax, cross_entropy

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)

        )

    def forward(self, image):
        return self.seq(image)

    def predict(self, image):
        x = self.forward(image)
        x = softmax(x, dim=1)
        _, preds = t.max(x, dim=1)
        return preds

    def accuracy(self, x, y):
        return t.sum(x == y).item() / len(x)


model = cnn()

loss_fn = cross_entropy
lr = 1e-3                                                                    # h2
opt = t.optim.SGD(model.parameters(), lr=lr)

# training
x_axis = []
y_axis = []


def fit(num_epochs):
    for epoch in range(num_epochs):
        for images, labels in train:
            x = model.forward(images)
            loss = loss_fn(x, labels)
            loss.backward()
            opt.step()
            opt.zero_grad()

        x_axis.append(epoch + 1)
        acc_per_batch = []
        with t.no_grad():
            for a, b in val:
                acc_ = model.accuracy(model.predict(a), b)
                acc_per_batch.append(acc_)
        y_axis.append(sum(acc_per_batch) / len(acc_per_batch))
        print(epoch+1, '-----', sum(acc_per_batch) / len(acc_per_batch))
fit(15)
t.save(model.state_dict(), 'cifar10.pth')
