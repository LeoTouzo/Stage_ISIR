import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# from torch.autograd import Variable
# from PIL import ImageFile
from dataloader import *
from plotdata import *
# import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

learning_rate = 0.01
momentum = 0.5
batch_size_train = 40
batch_size_test = 500


# evaluation on a batch of test data:
def evaluate(model, data):
    batch_enum = enumerate(data.loader_test)
    batch_idx, (testdata, testtargets) = next(batch_enum)
    model = model.eval()
    oupt = torch.argmax(model(testdata), dim=1)
    t = torch.sum(oupt == testtargets)
    result = t * 100.0 / len(testtargets)
    model = model.train()
    print(f"{t} correct on {len(testtargets)} ({result.item()} %)")
    return result.item()


# iteratively train on batches for one epoch:
def train_epoch(model, optimizer, data):
    batch_enum = enumerate(data.loader_train)
    i_count = 0
    iterations = data.num_train_samples // data.batch_size_train
    for batch_idx, (dt, targets) in batch_enum:
        dt.to(device)
        targets.to(device)
        i_count = i_count+1
        outputs = model(dt)
        loss = F.cross_entropy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i_count % 30:
            print(f"    step {i_count} / {iterations}")
        if i_count == iterations:
            break


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(((((75-2)//2-2)//2)**2)*64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x.view(-1, 3, 75, 75)))
        x = self.dropout1(F.max_pool2d(x, 2))
        x = F.relu(self.conv2(x))
        x = self.dropout2(F.max_pool2d(x, 2))
        x = torch.flatten(x, 1)
        x = self.dropout3(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# data = loadImgs(batch_size_train=batch_size_train, batch_size_test=batch_size_test)

net = Net().to(device)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
#
net.load_state_dict(torch.load('./data/model_TP.pt'))
net.eval()
#
# num_epochs = 100
# epochs = np.arange(num_epochs+1)
# accuracy = [evaluate(net, data)]
# for j in range(num_epochs):
#     print(f"epoch {j} / {num_epochs}")
#     train_epoch(net, optimizer, data)
#     accuracy.append(evaluate(net, data))
#     torch.save(net.state_dict(), './data/model_TP.pt')
#
# # plt.plot(epochs,accuracy)
# # plt.xlabel('epoch')
# # plt.ylabel('accuracy')
# np.savetxt('.data/accuracy.txt',accuracy)
#
# indices = np.random.choice(range(data.num_test_samples), 20)
# #
# plotdata(data, indices, net, original=False)

# evaluate(net,data)

def several_crops(model, n_crops, des_dir="./data/", img_size=100, batch_size_test=100):
    model = model.eval()
    oupt = []
    for i in range(n_crops):
        dataset_test = dset.ImageFolder(root=des_dir + "test/",
                                        transform=transforms.Compose([
                                            transforms.Resize(img_size),
                                            transforms.RandomCrop(75, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]))
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False)
        batch_enum = enumerate(dataloader_test)
        batch_idx, (testdata, testtargets) = next(batch_enum)
        oupt.append(model(testdata))
    avg_oupt = torch.argmax(sum(oupt), dim=1)
    t = torch.sum(avg_oupt == testtargets)
    result = t * 100.0 / len(testtargets)
    model = model.train()
    print(f"{t} correct on {len(testtargets)} ({result.item()} %)")
    return result.item()

# several_crops(net, 5, batch_size_test=batch_size_test)

def worst(n, indices, model, data):
    model = model.eval()
    testdata_full = data.test
    testdata = torch.stack([testdata_full[i][0] for i in indices])
    testtargets = torch.tensor([testdata_full[i][1] for i in indices])
    oupt = model(testdata)
    pwrong = torch.exp(oupt[torch.arange(oupt.size(0)),1-testtargets])/torch.sum(torch.exp(oupt),dim=1)
    argworst = torch.topk(pwrong, n)
    print(testtargets[argworst.indices])
    plt.figure()
    plotdata(data, argworst.indices, net, original=False)
    return argworst

# worst(5, np.arange(200), net, data)

x = torch.rand((3,75,75))*1e-1
x.requires_grad = True
for param in net.parameters():
    param.requires_grad = False

lr = 0.01
num_epochs = 100000
loss = nn.Softmax()(net(x)[0])[1]
epochs = np.arange(num_epochs+1)
loss_list = [loss.item()]
for j in range(num_epochs):
    loss.backward()
    with torch.no_grad():
        x -= lr*x.grad
        x.grad.zero_()
    loss = nn.Softmax()(net(x)[0])[1]
    loss_list.append(loss.item())
    if j%1000==0:
        print(j)

plt.plot(epochs,loss_list)
np.savetxt('.data/young_loss.txt',loss_list)
plt.xlabel('epoch')
plt.ylabel('P(class=0)')
plt.figure()
print(nn.Softmax()(net(x)))
im = np.transpose(x.detach().numpy(),(1,2,0))
np.savetxt('./data/young.txt',im)
plt.imshow(im)
plt.show()