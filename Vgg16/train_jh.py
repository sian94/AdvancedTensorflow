"""
Created on Sat Mar. 09 15:09:17 2019

@author: JHY

main for VGG16 CIFAR-10

with weight decay and learning rate decay

"""

from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 


import argparse, utils_jh, model_jh
import numpy as np

FLAGS = None

def get_arguments():
    
    parser = argparse.ArgumentParser('Implementation for MNIST handwritten digits 2020')
    
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=300,
                        help='Parameter for learning rate', 
                        required = False)
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128,
                        help='parameter for batch size', 
                        required = False)
    
    parser.add_argument('--eval_size', 
                        type=int, 
                        default=1000,
                        help='parameter for batch size', 
                        required = False)
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.001,
                        help='Parameter for learning rate', 
                        required = False)
    
    parser.add_argument('--net_type', 
                        type=str, 
                        #default='Dense',
                        #default='Conv',
                        default = 'Vgg16',
                        help='Parameter for Network Selection', 
                        required = False)
    
    return parser.parse_args()

    
def main():
    
    args = get_arguments()
    cfg = utils_jh.Config(args)
    
    print("---------------------------------------------------------")
    print("         Starting CIFAR Batch Processing Example")
    print("---------------------------------------------------------")

    cifar10 = utils_jh.Cifar10Reader(cfg)
    
    net = model_jh.Vgg16(cfg)
    
    if cfg.net_type == 'Vgg16':
        net = model_jh.Vggg16(cfg)
        wd_rate = 5*(1e-4)
    elif cfg.net_type == 'Resnet':
        wd_rate = 1*(1e-4)
    
    _train_op, _losses, _logits = net.optimizer()
    
    
    per_epoch = cifar10.image_size // cfg.batch_size
    
    print(per_epoch)
    
    learning_rate = cfg.learning_rate
    max_accuracy = 0
    
    for epoch in range(args.num_epoch):
        mean_loss = 0
        for steps in range(per_epoch):
            images, labels = cifar10.next_batch()
            
            fd=  {net.batch_img : images, net.batch_lab: labels, net.lr_decay: learning_rate, net.wd_rate : wd_rate}
            
            _, loss = net.sess.run((_train_op, _losses), feed_dict = fd)
            
            mean_loss += loss
            
        mean_loss /= float(per_epoch)
        print('epoch %d'%epoch, '======= Cost of %1.8f'%mean_loss)
        
        ########################################
        #EVALUATION
        #####################
        total_accuracy =0
        for step in range(cifar10.eval_num): # Eval data의 batch 개수
            eval_image, eval_label = cifar10.eval_batch()
            
            fd  = {net.batch_img : eval_image, net.batch_lab : eval_label}
            logits = net.sess.run((_logits), feed_dict = fd)
            
            correct_prediction = np.equal(np.argmax(logits, axis=1), eval_label)
            total_accuracy += np.mean(correct_prediction)
        
        total_accuracy /= cifar10.eval_num
        print("Evaluation at %d epoch"%epoch, "====== Accuracy of %1.4f"%total_accuracy)
        
        if max_accuracy > total_accuracy:
            learning_rate /= 10.
        else:
            max_accuracy = total_accuracy
        
   
if __name__ == '__main__':
       
    main() 
