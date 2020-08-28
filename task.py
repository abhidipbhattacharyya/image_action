import torch.nn as nn
import torch
from transformers import *
#from dataloader import *
from pascal_voc_dataset import *
from model import *
from loss import *
from torch.autograd import Variable
import argparse
import os
import collections
import random
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from evaluation import *
from optimization import BertAdam

class MIL:
    def __init__(self, args):
        self.args = args
        self.raw_data = Raw_dataset(args.flickerfile, args.label_list)
        self.transformer = get_transformer(network=args.network)
        self.evaluator = evaluator(self.raw_data.label_list )
        shuffle = False
        if args.mode !='test':
            shuffle = True

        self.model = ActionClassifier(self.raw_data.get_num_of_labels(), head = args.head, pos_dim =4, hidden_size= 512, is_tsv_feat = False, in_channels = args.num_boxes )
        self.epoch = args.epochs
        self.save_epoch = args.save_epoch
        self.lr=args.learning_rate

        if args.mode == 'train'  or args.mode == 'small_data' :
            self.data_set = MIL_dataset(self.raw_data, self.transformer, img_path = args.img_root_dir, tsv_path = args.tsv_path, mode = args.mode, use_tsv = False, num_boxes = args.num_boxes)
            self.data_loader = DataLoader(self.data_set, batch_size = args.batch_size, shuffle = shuffle, num_workers = args.num_workers)
            #params = list(self.decoder.parameters())
            #params.extend(list(self.encoder.parameters()))

            batch_per_epoch = len(self.data_loader)
            t_total = int(batch_per_epoch * args.epochs)

        if args.mode == 'dev' :
            self.eval_set = MIL_dataset(self.raw_data , self.transformer , img_path = args.img_root_dir, tsv_path = args.tsv_path, mode = args.mode, use_tsv = False, num_boxes = args.num_boxes)
            self.eval_loader = DataLoader(self.eval_set, batch_size = args.batch_size, shuffle = shuffle, num_workers = args.num_workers)

        self.st_epoch  = 0
        if(args.load !=None):
            fname = os.path.join(args.model_dir, args.load)
            self.st_epoch = self.load_model(fname)

        self.model.cuda()
        if args.multiGPU:
            self.model = nn.DataParallel(self.model)


        self.criterion = Mixture_loss(args.head)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9, weight_decay = 0.1)

        if args.mode == 'train':
            batch_per_epoch = len(self.data_loader)
            print('batch per epoch ={}'.format(batch_per_epoch))
            t_total = int(batch_per_epoch * args.epochs)
            print('total iterations== {} ; warmup start = {}'.format(t_total, t_total*args.wstep))
            self.optimizer = BertAdam(list(self.model.parameters()),
                                 lr=args.learning_rate,
                                 warmup=args.wstep,
                                 t_total=t_total)#changing warmup from 0.1 to 0.3
    def train(self):
        print('training started')
        self.model.train()
        for epoch in range(self.epoch):
            tr_loss = 0
            nb_tr_steps = 0
            em_loss_t = 0
            cls_loss_t = 0
            for imgs, subimgs, boxes, interaction_pattern, label_hot_vec in tqdm(self.data_loader):
                self.optimizer.zero_grad()
                imgs, subimgs, boxes, interaction_pattern, label_hot_vec = imgs.cuda(), subimgs.cuda(), boxes.cuda(), interaction_pattern.cuda(), label_hot_vec.cuda()
                g_x, p_yi_x = self.model(imgs, subimgs, boxes, interaction_pattern, label_hot_vec)

                loss, em_loss, class_loss = self.criterion(g_x, p_yi_x,label_hot_vec)#(h_x, g_x, p_y_x, p_yi_x,label_hot_vec)

                if self.args.multiGPU:
                    loss = loss.mean()

                tr_loss += loss.item()
                em_loss_t += em_loss
                cls_loss_t +=class_loss

                nb_tr_steps += 1
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                #nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.)
                self.optimizer.step()

            print("Train loss@epoch {}: total:{} emloss:{}, cls_loss:{}".format(self.st_epoch + epoch+1, tr_loss/nb_tr_steps, em_loss_t/nb_tr_steps,cls_loss_t/nb_tr_steps))
            if epoch == self.epoch-1 or (epoch + 1)% self.save_epoch == 0:
                filename = 'pascal_voc'+str(self.st_epoch + epoch+1)+'.model'
                filename = os.path.join(args.model_dir,filename)
                self.save_model(filename, self.st_epoch + epoch+1)

    def evaluate(self, thresold = 0.5):
        print('evaluation started')
        self.model.eval()
        res = []
        gold = []
        for imgs, subimgs, boxes, interaction_pattern, label_hot_vec in tqdm(self.eval_loader):
            imgs, subimgs, boxes, interaction_pattern =imgs.cuda(), subimgs.cuda(), boxes.cuda(), interaction_pattern.cuda()
            g_x, p_yi_x = self.model(imgs, subimgs, boxes, interaction_pattern)
            g_x = g_x.unsqueeze(dim =-1)
            class_prob = torch.bernoulli(p_yi_x) * g_x
            class_prob = torch.sum(class_prob,dim=1) - thresold
            class_prob = class_prob.cpu()
            target = [torch.nonzero(t).squeeze(-1).numpy() for t in label_hot_vec]
            pred = [(t>0).nonzero().squeeze(-1).numpy() for t in class_prob]

            for gs in target:
                gold.append(gs)

            for pd in pred:
                res.append(pd)

        self.evaluator.evaluate(res, gold,self.args.dump)


    def evaluate2(self, thresold = 0.5):
        print('evaluation started')
        self.model.eval()
        res = []
        gold = []
        for imgs, subimgs, boxes, interaction_pattern, label_hot_vec in tqdm(self.eval_loader):
            imgs, subimgs, boxes, interaction_pattern =imgs.cuda(), subimgs.cuda(), boxes.cuda(), interaction_pattern.cuda()
            g_x, p_yi_x = self.model(imgs, subimgs, boxes, interaction_pattern)
            g_x = g_x.unsqueeze(dim =-1)
            #class_prob = torch.bernoulli(p_yi_x) * g_x
            #class_prob = torch.sum(class_prob,dim=1) - thresold
            #class_prob = class_prob.cpu()
            target = [torch.nonzero(t).squeeze(-1).numpy() for t in label_hot_vec]
            pred = (p_yi_x >= thresold).float() * 1
            pred = torch.prod(pred, dim = 1)
            #pred = torch.sum(pred, dim =1)

            pred1 = [(t>0).nonzero().squeeze(-1).numpy() for t in pred]
            #pred1 = [(t>9).nonzero().squeeze(-1).numpy() for t in pred] #atleast 9 distribution among 16 says yes
            for gs in target:
                gold.append(gs)

            for pd in pred1:
                res.append(pd)

        self.evaluator.evaluate(res, gold,self.args.dump)


    def save_model(self, name, epoch):
        #epoch = self.epoch
        lr = self.lr
        check_point = {}
        check_point['model'] = self.model.state_dict()
        #check_point['decoder'] = self.decoder.state_dict()
        check_point['epoch'] = epoch
        check_point[ 'lr'] = lr
        check_point ['optimizer']  = None
        torch.save(check_point, name)
        print('model saved at {}'.format(name))

    def load_model(self, path):
        print("Load model from %s" % path)
        check_point= torch.load(path)
        model_dict = check_point['model']
        #decoder_dict = check_point['decoder']

        self.model.load_state_dict(model_dict , strict=False)
        #self.decoder.load_state_dict(decoder_dict, strict=False)
        if check_point ['optimizer']!=None:
            self.optimizer = optimizer

        return check_point['epoch']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Action Detection')
    parser.add_argument('--flickerfile', type=str, default = '/media/abhidip/2F1499756FA9B115/data/pascal_voc/VOCdevkit/VOC2012/ImageSets/Action/train.json')#'/media/abhidip/2F1499756FA9B115/data/flickr/abhidip_splits/flickrdata_VNPB_tfIdf2.json')
    parser.add_argument('--label_list', type = str, default = '/home/abhidip/projects/image_actions/image_action_gau/label_list_pvoc.txt')#label_list_Prop.txt
    parser.add_argument('--img_root_dir', type = str, default = '/media/abhidip/2F1499756FA9B115/data/pascal_voc/VOCdevkit/VOC2012/action_images')#'/media/abhidip/2F1499756FA9B115/data/flickr/flickr30k-images')
    parser.add_argument('--tsv_path', type = str, default = '/media/abhidip/2F1499756FA9B115/data/pascal_voc/VOCdevkit/VOC2012/code')#'/media/abhidip/2F1499756FA9B115/data/flickr/image_feat')

    parser.add_argument('--mode', choices=['train', 'dev', 'test', 'small_data'], type = str, default = 'train')
    parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161'], default='vgg19',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=4877, help='random seed')

    parser.add_argument("--batch_size",dest='batch_size', type=int, default=32)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)
    # Model parameters

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument("--wstep",type=float, default=0.1, help = 'warm up step with resect to toal iteration. Default 0.1')
    parser.add_argument('--model_dir', type=str, default="/media/abhidip/2F1499756FA9B115/iamge_action/Gauss1", help='model name to be saved')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--dump', type=str, default=None, help='path to save the output')

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=True, const=True)
    parser.add_argument('--head', type=int, default=8)# 16
    parser.add_argument('--num_boxes', type=int, default= 5)#None


    parser.add_argument("--is_tsv_feat", type=bool ,default = False)
    args = parser.parse_args()

    mil = MIL(args)

    if args.mode == 'train' or args.mode == 'small_data':
        mil.train()
    else:
        mil.evaluate()
