import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models import densenet161, resnet152, vgg19
import torchvision.transforms as transforms
from model_component import *


class ActionClassifier(nn.Module):
    def __init__(self, num_of_class, head = 16, pos_dim =4, hidden_size= 512, is_tsv_feat = False, in_channels = 36):
        super(ActionClassifier, self).__init__()
        self.interaction_network = Interaction_Network(in_channels = in_channels)
        self.image_encoder = Encoder() # either use same encoder for both image and object
        self.object_encoder = Object_Encoder()# use different to catch the global vs local features
        self.self_attention = MultiHeadedAttention2(head, self.image_encoder.dim) # attend self features..
        self.layer_norm1 = torch.nn.LayerNorm (self.image_encoder.dim)
        self.global_attention =MultiHeadedAttention(head, self.image_encoder.dim)#obj, interaction and fullimage
        self.layer_norm2 = torch.nn.LayerNorm (self.image_encoder.dim)
        self.classifier = Classifier(num_of_class, num_of_mixture = head, num_obj = in_channels)#DNN for classification
        self.classifier = self.classifier.double()
        self.is_tsv_feat = is_tsv_feat
        if is_tsv_feat:
            self.vis_encoder = VisualFeatEncoder(feat_dim = 2048, pos_dim = 4, hidden_size = self.image_encoder.dim, is_tsv = is_tsv_feat)

    def forward(self, image, objects, boxes, interaction_pattern, label = None):

        int_patt = self.interaction_network(interaction_pattern)
        img = self.image_encoder(image)

        if self.is_tsv_feat == False:
            obj_feat = torch.zeros(objects.size(0),objects.size(1), self.object_encoder.dim)
            ##print('obj shape b4 {}'.format(objects.size()))
            for i in range(objects.size(1)):
                obj_feat[:,i,:] = self.object_encoder(objects[:,i,:,:,:], boxes[:,i,:]) #batch x ith subImg x ch x h x w

            #print('obj shape after {}'.format(obj_feat.size()))
        else:
            obj_feat = self.vis_encoder((objects, boxes))

        obj_feat = obj_feat.cuda()

        #print('box shape {}'.format(boxes.size()))
        print('obj shape {}'.format(obj_feat.size()))
        print('int_patt shape {}'.format(int_patt.size()))
        #print('img shape {}'.format(img.size()))

        self_att_feat = self.self_attention(obj_feat, obj_feat, obj_feat)
        self_att_feat =self.layer_norm1(self_att_feat + obj_feat )
        #layer norm for self_att_feat
        print('self att shape {}'.format(self_att_feat.size()))
        global_att_feat = self.global_attention(obj_feat,int_patt,img) #(int_patt, obj_feat, obj_feat)#
        global_att_feat = self.layer_norm2(global_att_feat)
        #layer norm for global_att_feat
        print('global att shape {}'.format(global_att_feat.size()))
        feat = (global_att_feat + self_att_feat)/2 #explore other option.
        print('feat=== {} '.format(feat.size()))
        if label  != None:
            g_x, p_y_x, h_x, p_yi_x = self.classifier(feat.double(), label)
            return g_x, p_y_x, h_x, p_yi_x
        else:
            g_x, p_yi_x = self.classifier(feat.double())
            return g_x, p_yi_x
