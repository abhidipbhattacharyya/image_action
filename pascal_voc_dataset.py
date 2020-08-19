import os
import cv2
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset
import skimage.io
import skimage.transform
import json
from tqdm import tqdm
from utils import load_obj_tsv
import math
import torch.nn.functional as F

class Raw_dataset:
    def __init__(self, pascalvoc_file, label_list,top_labels = 3):
        self.data = json.load(open(pascalvoc_file,'r'))
        self.label_list = []

        with open(label_list, 'r') as l:
            ll = l.readlines()

        self.label_list = [l.strip() for l in ll]

        #with open(label_list,)
        #self.label_list.append('UNK')
        self.label_map= {t:i for i,t in enumerate(self.label_list)}
        print(self.label_map)

    def label_to_id(self, label):
        if label not in self.label_map:
            return self.label_map['other']
        else:
            return self.label_map[label]

    def id_to_label (self, id):
        if id >= len(self.label_list):
            return 'other'
        return self.label_list[id]

    def get_label(self, label):
        if label not in self.label_map:
            return 'other'
        else:
            return label

    def get_num_of_labels(self):
        return len(self.label_list)


class MIL_dataset(Dataset):

    def __init__(self, dataset, transform, img_path = None, tsv_path = None, mode= 'train',use_tsv = False, num_boxes = None):
        '''
        dataset: rawdataset with flickr image and sentences
        img_path: path for images to be loaded from
        mode: 'training' or 'dev' or 'test'
        img_list_fname: the file containing the list of images to be fed to the system
        '''
        super().__init__()
        self.raw_dataset = dataset
        self.img_data_map = {dp['image']:dp for dp in self.raw_dataset.data}
        self.img_path =  os.path.join(img_path, mode)
        self.transform =  transform
        self.mode = mode
        self.tsv = use_tsv
        self.tsv_path = tsv_path

        tsv_file = os.path.join(self.tsv_path, '{}.tsv'.format(mode))
        img_data = load_obj_tsv(tsv_file, use_tsv = self.tsv)
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']+ '.jpg'] = img_datum

        self.num_boxes = num_boxes
        self.non_label = []

        self.features = self.convert_example_to_feature()

    def convert_single_example (self, datapoint):
        imagename = datapoint['image']

        label_hot_vec= [0]*self.raw_dataset.get_num_of_labels()

        for obj in datapoint["objects"]:
            for action in obj["action"]:
                label_hot_vec[self.raw_dataset.label_to_id(action)] = 1
        total = sum(label_hot_vec)
        #if no other label then UNK
        if total == 0:
            #self.data_stat.update(['UNK'])
            label_hot_vec[self.raw_dataset.label_to_id('UNK')] = 1
            total = 1
            self.non_label.append(datapoint)

        img_info = self.imgid2img[imagename]
        obj_num = img_info['num_boxes']
        if self.tsv:
            feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) #== len(feats)

        if self.num_boxes == None:
            self.num_boxes = obj_num
        else:
            self.num_boxes = min(obj_num,self.num_boxes )


        imagename = os.path.join(self.img_path, imagename )
        datapoint['label_hot_vec'] = label_hot_vec
        datapoint['boxes'] = boxes[:self.num_boxes]
        if self.tsv:
            datapoint['tsv_feat'] = feats[:self.num_boxes]
        datapoint['img_height'], datapoint['img_width'] = img_info['img_h'], img_info['img_w']
        datapoint['image'] =   os.path.join(self.img_path, imagename )

        #print('===={}'.format(datapoint['label_hot_vec']))

        return datapoint


    def convert_example_to_feature(self):
        features = []
        missing = 0
        print('processing for features')
        img_list = list(self.imgid2img.keys())
        for img in tqdm(img_list):
            if img in self.img_data_map:
                dp = self.img_data_map[img]
                feat = self.convert_single_example(dp)
                features.append(feat)
            else:
                missing = missing + 1

        print('missing data {}'.format(missing))
        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        f_i = self.features[index]
        imagename = f_i['image']
        image = Image.open(imagename)
        image = image.convert('RGB')
        #image_tensor = torch.FloatTensor(image)

        transformed_image = self.transform(image)

        #print('image shape {}'.format(image.size))
        #print('height = {}, width = {}'.format(f_i['img_height'], f_i['img_width']))

        subimages = []#*(self.num_boxes+1)
        #subimages.append(transformed_image)
        interaction_pattern = []#torch.zeros(self.num_boxes, f_i['img_height'],f_i['img_width'])

        for i,box in enumerate(f_i['boxes']):
            if self.tsv == False:
                image_i = image.crop(box)#[:,box[1]:box[3], box[0]:box[2]]
                image_i = torch.FloatTensor(self.transform(image_i))
                subimages.append(image_i)
                #print( image_i.size())
            in_pat_i = torch.zeros(1, f_i['img_height'],f_i['img_width'])
            in_pat_i [0, int(math.floor(box[1])): int(math.floor(box[3]))+1,  int(math.floor(box[0])): int(math.floor(box[2]))+1] = 1

            #print('b4  interp111 {}'.format(in_pat_i.size()))
            in_pat_i = in_pat_i.permute(0,2,1)
            in_pat_i = F.interpolate(in_pat_i, size=224)
            #print('after interp111 {}'.format(in_pat_i.size()))
            in_pat_i = in_pat_i.permute(0,2,1)
            in_pat_i = F.interpolate(in_pat_i, size=224)

            #print('after interp 222{}'.format(in_pat_i.size()))
            interaction_pattern.append(in_pat_i)

        interaction_pattern = torch.stack(interaction_pattern, dim =0)
        interaction_pattern = interaction_pattern.squeeze(1)

        boxes = f_i['boxes']
        boxes[:, (0, 2)] /= f_i['img_width']
        boxes[:, (1, 3)] /= f_i['img_height']
        if(len(subimages) > 0):
            subimages = torch.stack(subimages, dim =0)



        if self.tsv == True:
            if mode == 'test':
                return torch.FloatTensor(transformed_image ), torch.FloatTensor(f_i['tsv_feat']), torch.FloatTensor(boxes),  torch.FloatTensor(interaction_pattern),torch.FloatTensor([])
            else:
                return torch.FloatTensor(transformed_image ), torch.FloatTensor(f_i['tsv_feat']), torch.FloatTensor(boxes), torch.FloatTensor(transformed_image ), torch.FloatTensor(interaction_pattern),torch.FloatTensor(f_i['label_hot_vec'])
        else:
            if self.mode == 'test':
                return  torch.FloatTensor(transformed_image ),torch.FloatTensor(subimages), torch.FloatTensor(boxes),torch.FloatTensor(interaction_pattern), torch.FloatTensor([])
            else:
                    #print('I returned')
                    #print(torch.FloatTensor(transformed_image).size())
                    #print(torch.FloatTensor(subimages).size())
                    #print(torch.FloatTensor(boxes).size())
                    #print(torch.FloatTensor(interaction_pattern).size())
                    #print(torch.FloatTensor(f_i['label_hot_vec']).size())
                return  torch.FloatTensor(transformed_image ),torch.FloatTensor(subimages), torch.FloatTensor(boxes), torch.FloatTensor(interaction_pattern), torch.FloatTensor(f_i['label_hot_vec'])
