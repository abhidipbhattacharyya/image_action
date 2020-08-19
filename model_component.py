import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import densenet161, resnet152, vgg16
import torchvision.transforms as transforms
import math, copy, time
import numpy as np

def get_transformer(network = 'vgg19'):
    '''
    get appropriate tranformer for preprocessing
    '''
    size =224
    if network =='vgg19' or network == 'resnet':
        size = 224
    elif network == 'inception_v3':
        size = 299

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return transform

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadedAttention2(nn.Module):
    def __init__(self,h, d_model, dropout=0.1):
        super(MultiHeadedAttention2, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attention = nn.MultiheadAttention(d_model, h, dropout)

    def forward(self,query, key, value):
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        attn_output, attn_output_weights = self.attention(query, key, value)
        return attn_output

class VisualFeatEncoder(nn.Module):
    def __init__(self, feat_dim = 2048, pos_dim = 4, hidden_size = 768, is_tsv = True, hidden_dropout_prob = 0.1 ):
        super().__init__()
        feat_dim = feat_dim
        pos_dim = pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, hidden_size)
        self.visn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, hidden_size)
        self.box_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, visn_input):
        feats, boxes = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output

class Object_Encoder(nn.Module):
    def __init__(self, network='vgg16', pretrained=True, hidden_dropout_prob = 0.1):
        super(Object_Encoder, self).__init__()
        self.net = vgg16(pretrained= pretrained)
        self.net = nn.Sequential(*list(self.net.features.children()))
        layers = []
        conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.feat = nn.Sequential(*layers)
        self.dim = 512


        self.visn_layer_norm = nn.LayerNorm(self.dim, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(4, self.dim)
        self.box_layer_norm = nn.LayerNorm(self.dim, eps=1e-12)

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x, boxes):
        #with torch.no_grad():
        x = self.net(x)
        #print('ob En ====={}'.format(x.size()))
        x = self.feat(x)
        x = x.permute(0, 2, 3, 1)
        #print('ob En ====={}'.format(x.size()))
        x = x.contiguous()
        x = x.view(x.size(0), -1, x.size(-1))
        x = x.squeeze(dim =1)
        #print(x)
        #print('ob ====={}'.format(x.size()))
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output
        #return x

class Encoder(nn.Module):
    def __init__(self, network='vgg16', pretrained=True):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'densenet161':
            self.net = densenet161(pretrained=True)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        else:
            self.net = vgg16(pretrained= pretrained)
            self.net = nn.Sequential(*list(self.net.features.children()))
            #layers = [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
            #layers+= [nn.ReLU(inplace=True)]
            #layers+= [nn.MaxPool2d(kernel_size=2, stride=2)]
            #self.layers =  nn.Sequential(*layers)
            self.dim = 512

        self.dense = nn.Linear(196, 128)

    def forward(self, x):
        #with torch.no_grad():
        x = self.net(x)

        #x = self.layers(x)
        x = x.permute(0, 2, 3, 1)
        #print('En ====={}'.format(x.size()))
        x = x.contiguous()
        x = x.view(x.size(0), -1, x.size(-1))
        #print(x)
        #print('====={}'.format(x.size()))
        return x

class Interaction_Network(nn.Module):
    def __init__(self, in_channels = 36, init_weights=True ):
        super(Interaction_Network ,self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.in_channels = in_channels
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        x = x.view(x.size(0), -1, x.size(-1))
        return x

    def make_layers(self, batch_norm=False):
        layers = []
        in_channels = self.in_channels
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



class Classifier(nn.Module):
    def __init__(self, num_classes, num_of_mixture = 8, input_dim = 512, output_dim = 256, num_obj = 36):
        super(Classifier, self).__init__()
        gx_dim  = output_dim * num_of_mixture
        self.num_of_mixture = num_of_mixture
        self.num_classes = num_classes
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_of_mixture ),
            nn.Conv1d(num_obj, 1 , 1)

        )
        self.proba = nn.Sequential(
            nn.Linear(input_dim , 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
            nn.Conv1d(num_obj, num_of_mixture, 1)
        )
    def forward(self,x, label = None):
        #x = torch.flatten(x, 1)
        #print(x.size())
        #x = x.double()
        g_x = self.gate(x).squeeze(dim=1)
        #print('gx shape {}'.format(g_x.size()))
        g_x = F.softmax(g_x, dim = -1) # bs x K
        p_yi_x = torch.sigmoid(self.proba(x)) # bs x num_of mix x num class
        #p_yi_x = torch.bernoulli(p_yi_x)
        if label == None:
            return g_x, p_yi_x

        else:
        #p_yi_x = p_yi_x.view(p_yi_x.size(0),-1,self.num_classes) # bs x K x num_of_class
            #print('p_yi_x shape {}'.format(p_yi_x.size()))
            with torch.no_grad():
                label = label.detach().unsqueeze(dim =1) #bs x 1 x num clas
                label = torch.repeat_interleave(label, repeats=self.num_of_mixture, dim=1) #bs x self.num_of_mixture x num clas
                label_c = 1 - label

            p_yi_x_c = 1 - p_yi_x
            #print('label shape {}'.format(label.size()))
            #print('label  {}'.format(label ))
            p_yi_x1 = p_yi_x * label + label_c * p_yi_x_c
            #print('pyix after {}'.format(p_yi_x))
            p_y_x = torch.prod(p_yi_x1, dim = -1 )#bs x num of mix
            #print('pyx shape {}'.format(p_y_x.size()))
            #print(p_y_x.type())
            h_x = g_x * p_y_x
            #print(h_x)
            h_x = F.softmax(h_x, dim = -1) #bs x num of mix

            #print('h_x {}'.format(h_x.size()))
            #print(h_x)
            return g_x, p_y_x, h_x, p_yi_x
