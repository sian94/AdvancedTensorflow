# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:11:06 2021

@author: Sian Yang
"""
import numpy as np
import os, cv2

class Config(object):
    
    cifar10_data = {'data_path': 'C:\\Coding\\data\\CIFAR\\CIFAR-10', 'train_name' : ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'],
                    'test_name' : 'test_batch'}
    
    
    
    def __init__(self, args):
        
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.eval_size = args.eval_size
        
        self.learning_rate = args.learning_rate
        self.net_type = args.net_type
        


class Cifar10Reader(object):
    
    def __init__(self, cfg):

        images = []
        labels = []
        for i in range(len(cfg.cifar10_data['train_name'])):
            train = self._unpickle(os.path.join(cfg.cifar10_data['data_path'], cfg.cifar10_data['train_name'][i]))
            images.append(train['data'])
            labels.append(train['labels'])
            
        
        self.images = np.reshape(images, (-1,32*32*3))
        self.labels = np.reshape(labels, (-1,))
        self.image_size = len(self.images)
        self.label_size = len(self.labels)
        print('image data size - ', self.image_size)

        test = self._unpickle(os.path.join(cfg.cifar10_data['data_path'], cfg.cifar10_data['test_name']))
        
        self.test_images = test['data']
        self.test_labels = test['labels']
        self.eval_size = cfg.eval_size
        self.eval_num = len(self.test_images)//cfg.eval_size
        self.eval_pos =0
        print('evaluation data size - ', len(self.test_images))
        
        self.batch_size = cfg.batch_size
        self.index = np.arange(self.image_size)
        self._shuffle()
        
        self.au_max_scale = 1.2
        self.au_min_scale = 0.8
            
    
    def _unpickle(self, file):
        
        import pickle 
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo, encoding='latin1')
            
        return dictionary
    
    def _shuffle(self):
        
        np.random.shuffle(self.index)
        self.position = 0
        
    def next_batch(self):
        
        images = []
        labels = []
        for batch in range(self.batch_size):
            image = np.transpose(np.reshape(self.images[self.index[self.position]], (3,-1))) # shape should be (1024,3)
            images.append(np.reshape(image, (32,32,3)))
            labels.append(self.labels[self.index[self.position]])
            
            self.position +=1
            
            if self.position == self.image_size:
                self._shuffle()
                
        # RESHAPE to (Batch size, 32, 32, 3)
        batch_image = np.reshape(images, (-1,32,32,3))
        batch_label = np.reshape(labels, (-1,))
        
        return batch_image, batch_label
    
    def eval_batch(self):
        images = []
        labels = []
        for batch in range(self.eval_size):
            image = np.transpose(np.reshape(self.test_images[self.eval_pos], (3,-1))) # shape should be (1024,3)
            images.append(np.reshape(image,(32,32,3)))
            labels.append(self.test_labels[self.eval_pos])
            
            self.eval_pos +=1
            
            if self.eval_pos == len(self.test_images):
                self.eval_pos =0
                
        batch_image = np.reshape(images, (-1,32,32,3))
        batch_label = np.reshape(labels, (-1,))
        return batch_image, batch_label
    
    def _img_aug(self, batch):
        
        # scale +- 20% (max_scalee=1.2, min_scale=0.8)
        random_scale = np.reshpae((self.au_max_scale - self.au_min_scale)
                                  * np.random.random(size=self.batch_size)+self.au_min_scale, (-1,1)) # 0.8~1.2 random numbers of shape (batch_size,)
        original_size = np.reshape(np.shape(batch)[1:3], (1,-1)) # SHAPE : (1,2)
        
        new_size = np.matmul(random_scale, original_size).astype(np.int32) #shape : (N,2)
        
        changes = new_size - original_size # shape : (N, 2)
        token = changes[:,0] >=0  # 어차피 H와 W는 같은 크기로 변하므로 하나만 갖고 옴. 
                            # meaning : target size is greater than original?to crop OR to pad
                            # shape : (N,) 
        changes = np.abs(changes)
        
        for idx in range(len(batch)): # for number of batches
            #flip horizontal
            if np.random.random()>0.5:
                batch[idx] = cv2.flip(batch[idx],1) # 1 means horizontal flip
                
            
        for idx in range(len(batch)):
            # resize image
            temp = cv2.resize(batch[idx], tuple(np.flip(new_size[idx])), interpolation=cv2.INTER_LANCZOS4) 
            
            #size up conversion (CROP)
            if token[idx]:
                crop_pos = np.random.random()*changes[idx] # random (0~1)*pixel changes(2,)
                crop_pos = crop_pos.astype(np.int32) # starting position(upper left)
                end_pos = crop_pos + original_size[0] # end position (lower right)
                                    #shape : (2,)
                
                batch[idx] = temp[crop_pos[0]:end_pos[0], crop_pos[1]:end_pos[1],:]
                                    #starting H,           starting             channel3
                
            
            #size down conversion (PAD)
            else: 
                pad_s = int(np.random.random()*changes[idx][0]) # random(0~1)*pixel changes in H
                pad_h = (pad_s, changes[idx][0]-pad_s)
                
                pad_s = int(np.random.random()*changes[idx][1]) # random(0~1)*pixel changes in W
                pad_w = (pad_s, changes[idx][1]-pad_s)
                
                batch[idx] = np.pad(temp, (pad_h, pad_w,(0,0)), mode='edge') # padding using 'edge' pixel values
            
            return batch
       
