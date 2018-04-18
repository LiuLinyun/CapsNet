import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import math

class Capsule(nn.Module):
    def __init__(self, input_features, output_features, input_feature_length, output_feature_length, routing_iterators):
        super(Capsule,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.input_feature_length = input_feature_length
        self.output_feature_length = output_feature_length
        self.routing_iterators = routing_iterators
        self.weight = nn.Parameter(torch.Tensor(input_features*output_features, output_feature_length, input_feature_length))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.size(0)
        b = Variable(torch.zeros(batch_size, self.input_features, self.output_features), requires_grad = False)
        if input.is_cuda:
            b = b.cuda()
        input = input.unsqueeze(dim=2).expand(-1,-1,self.output_features,-1).contiguous().view(batch_size, self.input_features*self.output_features, self.input_feature_length, 1)
        hat_u = torch.matmul(self.weight, input).view(batch_size,self.input_features,self.output_features,self.output_feature_length)
        for r in range(self.routing_iterators):
            c = F.softmax(b.view(batch_size, self.input_features, self.output_features), dim=-1).view(batch_size, self.input_features, self.output_features)
            hat_u_ = torch.mul(c.view(-1,1).expand(-1,self.output_feature_length), hat_u.view(-1, self.output_feature_length)).view(batch_size, self.input_features, self.output_features, self.output_feature_length)
            s = torch.sum(hat_u_, dim=1)
            s = s.view(batch_size*self.output_features, self.output_feature_length)
            s_norm = torch.norm(s, p=2, dim=1)
            s_norm_ = torch.div(s_norm, s_norm.pow(2).add(1))
            v = torch.mul(s_norm_.view(-1,1).expand(-1, self.output_feature_length), s)
            v = v.view(batch_size, self.output_features, self.output_feature_length)
            if r == self.routing_iterators-1:
                return v
            v = v.expand(self.input_features, batch_size, self.output_features, self.output_feature_length).transpose(0,1)
            b = b.add(torch.mul(hat_u, v).sum(-1))
        return v
