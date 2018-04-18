import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
import Capsule
import os

traindata = datasets.MNIST("./", train=True,
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,)), ]),
                            target_transform=None, download=False)
train_loader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=8)

testdata = datasets.MNIST("./", train=False,
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,)), ]),
                            target_transform=None, download=False)
test_loader = torch.utils.data.DataLoader(traindata, batch_size=16, shuffle=True, num_workers=8)

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1,256,9)
        self.conv2 = nn.Conv2d(256,8*32,9,2)
        self.relu = nn.ReLU()
        self.cap = Capsule.Capsule(input_features=32*6*6,output_features=10,input_feature_length=8,output_feature_length=16,routing_iterators=3)

    
    def forward(self, input):
        output = self.relu(self.conv1(input))
        output = self.conv2(output)
        # 把当前的输出从(batch_size, 8*32, 6, 6)整理成(batch_size, 32*6*6, 8)
        output = torch.transpose(output,1,-1).contiguous().view(output.size(0),32*6*6,8)
        output = self.cap(output)
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(16,512)
        self.linear2 = nn.Linear(512,1024)
        self.linear3 = nn.Linear(1024,784)

    def forward(self, input, capsoutput):
        _, unmasked_index = capsoutput.norm(p=2, dim=-1).max(dim=-1)
        if input.is_cuda:
            mask = Variable(torch.cuda.ByteTensor(capsoutput.size(0), capsoutput.size(1))).fill_(0).scatter_(1, unmasked_index.view(-1,1), 1)
        else:
            mask = Variable(torch.ByteTensor(capsoutput.size(0), capsoutput.size(1))).fill_(0).scatter_(1, unmasked_index.view(-1,1), 1)
        mask = mask.view(-1,1).expand(-1,capsoutput.size(2)).contiguous().view(capsoutput.size(0), capsoutput.size(1),capsoutput.size(2))
        unmasked_cap = capsoutput.masked_select(mask=mask).view(capsoutput.size(0),-1)
        output = self.relu(self.linear1(unmasked_cap))
        output = self.relu(self.linear2(output))
        output = self.sigmoid(self.linear3(output))
        return output

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.marginloss = nn.MultiMarginLoss(p=2, margin=0.9)
        self.mseloss = nn.MSELoss()

    def forward(self, input, target, capsoutput, decoderoutput):
        caploss = self.marginloss(capsoutput.norm(p=2, dim=-1), target)
        decloss = self.mseloss(decoderoutput, input.view(input.size(0), -1))
        sumloss = (1-0.0005)*caploss + 0.0005*decloss
        return sumloss

class WholeModel(nn.Module):
    def __init__(self):
        super(WholeModel,self).__init__()
        self.capsnet = CapsNet()
        self.decoder = Decoder()
        self.totalloss = TotalLoss()

    def forward(self, input, target):
        capsoutput = self.capsnet(input)
        decoderoutput = self.decoder(input, capsoutput)
        sumloss = self.totalloss(input, target, capsoutput, decoderoutput)
        return sumloss


def train(iterators, model_file, use_cuda=True, retrain=False):
    model = WholeModel()
    if not retrain:
        model.load_state_dict(torch.load(model_file))

    if use_cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for t in range(iterators):
        input, target = next(iter(train_loader))
        input = Variable(input)
        target = Variable(target)
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        loss = model(input, target)
        loss.backward()
        optimizer.step()
        print("train iters: " + str(t) + "\t loss: " + str(loss.cpu().data))
        if (t+1)%100 == 0:
            torch.save(model.state_dict(), model_file)

def test(model, input):
    model.train(False)
    _, output = model.capsnet(input).norm(p=2, dim=-1).max(dim=-1)
    return output

###############################train & test####################################
train(100, "minist.mod", use_cuda=True, retrain=False)
input, target = next(iter(test_loader))
input = Variable(input)
target = Variable(target)
model = WholeModel()
model.load_state_dict(torch.load("minist.mod"))
output=test(model, input)
print(target.view(1,-1))
print(output.view(1,-1))
