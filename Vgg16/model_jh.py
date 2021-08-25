# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:40:29 2021

@author: JHY
"""
from network import Network
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

class Vgg16(Network):
    
    def __init__(self, cfg):
        
        self.batch_img = tf.placeholder(dtype=tf.uint8)
        self.batch_lab = tf.placeholder(dtype=tf.uint8)
        self.labels = tf.cast(self.batch_lab, tf.int32)
        
        self.lr_decay = tf.placeholder(dtype=tf.float32)
        self.wd_rate = tf.placeholder(dtype=tf.float32)

        self.reservoir = {}
                
        self.cfg = cfg ###
        
        super(Vgg16, self).__init__()
    
        
    def _inference(self, tinput):
        
        inputs = tf.cast(tinput, tf.float32)/255. -0.5  #-0.5~-0.5
        
        (self.feed(inputs)
             .conv(lfilter_shape=(3,3,3,64), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal', sname = 'conv1')
             .conv(lfilter_shape=(3,3,64,64), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal',sname = 'conv2')
             .maxpool(sname = 'maxpool1')
             .conv(lfilter_shape=(3,3,64,128),lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal', sname = 'conv3')
             .conv(lfilter_shape=(3,3,128,128),lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal', sname = 'conv4')
             .maxpool(sname = 'maxpool2')
             .conv(lfilter_shape=(3,3,128,256),lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal', sname = 'conv5')
             .conv(lfilter_shape=(3,3,256,256), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal',sname = 'conv6')
             .conv(lfilter_shape=(3,3,256,256),lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal', sname = 'conv7')
             .maxpool(sname = 'maxpool3')
             .conv(lfilter_shape=(3,3,256,512), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal',sname = 'conv8')
             .conv(lfilter_shape=(3,3,512,512), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal',sname = 'conv9')
             .conv(lfilter_shape=(3,3,512,512), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal',sname = 'conv10')
             .maxpool(sname = 'maxpool4')
             .conv(lfilter_shape=(3,3,512,512), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal',sname = 'conv11')
             .conv(lfilter_shape=(3,3,512,512), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal',sname = 'conv12')
             .conv(lfilter_shape=(3,3,512,512), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal',sname = 'conv13')
             .conv(lfilter_shape=(2,2,512,512), lstrides=(1,1,1,1), spadding='VALID', buse_bias=False, sactivation='ReLu',  sinitializer='he_normal', sname = 'FCL1')
             .conv(lfilter_shape=(1,1,512,10), lstrides=(1,1,1,1), spadding='VALID', buse_bias=True, sactivation='None',  sinitializer='he_normal', sname = 'output'))
        ## 마지막 두줄
        
        return self.terminals[0]
    
    def _build(self):
        
        self.reservoir['logits'] = tf.reshape(self._inference(self.batch_img), (-1,10)) ## Nx1x1x10 >> N x10 reshape
        
        self.reservoir['ce_loss'] = self._createloss(name='loss')
    
    def _createloss(self, name=None, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse=reuse):
            
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels, logits = self.reservoir['logits'])
            loss = tf.reduce_mean(xentropy)
            
        return loss
    
    def optimizer(self):
        
        ####OPTIMIZERS
        weight_norm = self.wd_rate * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        total_loss = self.reservoir['ce_loss'] + weight_norm
        
        
        opt = tf.train.AdamOptimizer(learning_rate = self.lr_decay)
        train_op = opt.minimize(total_loss)
        
        ####SESSION CREATE
        gpu_options = tf.GPUOptions(allow_growth = True, allocator_type = 'BFC')
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        
        return train_op, total_loss , self.reservoir['logits']
        
            