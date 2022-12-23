import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import sys
import os
import torch_geometric as tg
from convlstm import ConvLSTM
torch.set_printoptions(threshold=sys.maxsize)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adj = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0., -1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
          1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,
          1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
          0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  1.,  0.]],dtype=torch.float32)

cls =torch.tensor([[0,	0,	0,	1,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	1,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	0,	0,	1,	0,	0],
[0,	-1,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	1,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	1,	0,	0,	0],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
[0,	0,	0, -1,	0,	0, 0,	0,	0,	0],
[0,	0,	-1,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	0,	0,	1,	0,	0],
[0,	0,	0,	0,	0,	0,	1,	0,	0,	0],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	1],
[0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	1,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	-1],
[0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
[0,	0,	0,	0,	-1,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0, 0,	0,	0,	0,	1],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	1,	0,	0,	0,	0,	0,	0],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0]],dtype=torch.float32)

adj2 = torch.tensor([[0,	-1,	-1,	1,	1,	1,	0,	0,	0,	0],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0,],
[1,	0,	0,	0,	0,	0,	1,	0,	0,	0,],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0,],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0,],
[0,	0,	1,	0,	-1,	0,	0,	0,	0,	1,],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0,],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0,],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0,],
[0,	0,	0,	0,	1,	0,	0,	0,	0,	0,]],dtype=torch.float32)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, mean=0, std=math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unit_gcn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.sig = nn.Sigmoid()
        self.adj = Variable(adj, requires_grad=False)
        self.mask = torch.nn.Parameter(torch.ones(35,35))
        self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        self.gcn = tg.nn.DenseGCNConv(40,40)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv, 1)

    def forward(self, x, d):
        adj_mat = None
        self.adj = self.adj.cuda(x.get_device())
        adj_mat = self.adj[:,:]
        self.mask = self.mask.cuda(x.get_device())
        adj_mat = adj_mat*self.mask+d
        x = self.gcn(x,adj_mat)
        y = self.conv(x)
        y = self.bn(y)
        y += self.down(x)
        y = y*self.sig(y)

        return y

class unit_gcn2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(unit_gcn2, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.sig = nn.Sigmoid()
        self.A = Variable(torch.eye(10), requires_grad=False)
        self.adj = Variable(adj2, requires_grad=False)
        self.cls =  Variable(cls, requires_grad=False)
        self.mask = torch.nn.Parameter(torch.ones(10,10))
        self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        self.w = torch.nn.Parameter(torch.zeros(10,10)+1e-6)
        self.gcn = tg.nn.DenseGCNConv(40,40)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv, 1)

    def forward(self, x, d):
        adj_mat = None
        self.A =self.A.cuda(x.get_device())
        self.adj = self.adj.cuda(x.get_device())
        adj_mat = self.adj[:,:] + self.A[:,:]
        self.mask = self.mask.cuda(x.get_device())
        adj_mat = adj_mat*self.mask+d
        x = self.gcn(x,adj_mat)
        y = self.conv(x)
        y = self.bn(y)
        y += self.down(x)
        y = y*self.sig(y)

        return y


class STCL_res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STCL_res, self).__init__()
        self.gcn = unit_gcn(in_channels, out_channels)
        self.con1 = nn.LSTM(input_size=35,hidden_size=35,num_layers=3,bias=True,batch_first=True,dropout=0,bidirectional=False)
        self.sig = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1)
    def forward(self, x, d):
        x = self.gcn(x, d)
        B, C, V, H = x.size()
        x = x.reshape(B*C, V, H)
        x = torch.transpose(x,1,2)
        x,_ = self.con1(x)
        x = torch.transpose(x,1,2)
        x = x.reshape(B, C, V, H)
        x = self.bn(x)
        x = x*self.sig(x)
        return x

class STCL2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STCL2, self).__init__()
        self.gcn = unit_gcn2(in_channels, out_channels)
        self.con1 = nn.LSTM(input_size=10,hidden_size=10,num_layers=3,bias=True,batch_first=True,dropout=0,bidirectional=False)
        self.sig = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1)
    def forward(self,x, d):
        x = self.gcn(x, d)
        B, C, V, H = x.size()
        x = x.reshape(B*C, V, H)
        x = torch.transpose(x,1,2)
        x,_ = self.con1(x)
        x = torch.transpose(x,1,2)
        x = x.reshape(B, C, V, H )
        x = self.bn(x)
        x = x*self.sig(x)
        return x

class Resnet1(nn.Module):
    def __init__(self,):
        super(Resnet1, self).__init__()
        self.l1 = STCL_res(64,64)
        self.l2 = STCL_res(64,128)
        self.l3 = STCL_res(128, 256)
        self.sig = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(256)
        self.down = nn.Sequential(
                nn.Conv2d(64, 256, 1),
                nn.BatchNorm2d(256)
            )
        bn_init(self.bn, 1)
    def forward(self, x, d):
        x1 = self.l1(x, d)
        x2 = self.l2(x1, d)
        x3 = self.l3(x2, d)
        a = self.bn(x3)
        a += self.down(x)
        a = a*self.sig(a)
        return a

class Resnet2(nn.Module):
    def __init__(self,):
        super(Resnet2, self).__init__()
        self.l1 = STCL2(256,64)
        self.l2 = STCL2(64, 128)
        self.l3 = STCL2(128, 256)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.glu = nn.GLU()
        self.sig = nn.Sigmoid()
        self.down = nn.Sequential(
                nn.Conv2d(256, 256, 1),
                nn.BatchNorm2d(256)
            )
        bn_init(self.bn, 1)
    def forward(self, x, d):
        x1 = self.l1(x, d)
        x2 = self.l2(x1, d)
        x3 = self.l3(x2, d)
        a = self.bn(x3)
        a += self.down(x)
        a = a*self.sig(a)
        return a

class FC1(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC1, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
        self.glu = nn.GLU(dim = 1)
        self.tan = nn.Tanh()
        bn_init(self.bn, 1)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / 3))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.fc(x)
        # x = x.view(x.size(0),1,x.size(1),1)
        # x = self.bn(x)
        # x = x.view(x.size(0),x.size(2))
        # x = x*self.sig(x)
        return x

def sof(x):
    x = torch.exp(x)
    x = x/torch.sum(x)
    return x

class Model(nn.Module):

    def __init__(self,):
        super(Model, self).__init__()

        self.l1 = unit_gcn(1,64)
        self.l2 = Resnet1()
        self.l3 = Resnet2()
        self.fc1 = FC1(256*40*45,20)
        self.drop = nn.Dropout(p=0.5)
        self.cls =  Variable(cls, requires_grad=False)
        self.w1 = nn.Linear(40,40)
        self.w2 = nn.Linear(40,40)
        self.w3 = nn.Linear(40,40)
        self.w4 = nn.Linear(40,40)
        self.tan = nn.Tanh()
        self.relu = nn.ReLU()
    def forward(self, x):
        cls =self.cls.cuda(x.get_device())
        cls = cls.t()
        d1 =self.tan(torch.matmul(self.relu(self.w1(x)),self.relu(torch.transpose(self.w2(x),2,3)))).reshape(35,35)
        a = torch.matmul(cls,x)
        d2 =self.tan(torch.matmul(self.relu(self.w1(a)),self.relu(torch.transpose(self.w2(a),2,3)))).reshape(10,10)
        x = self.l1(x,d1)
        x1 = self.l2(x,d1)
        x = torch.matmul(cls,x1)
        x2 = self.l3(x,d2)
        #x = torch.exp(x)
        x = torch.cat([x1,x2],2)
        bat,_,_,_ = x.size()
        x = x.view(bat,-1)
        x = self.drop(x)
        x = self.fc1(x)
        return x
