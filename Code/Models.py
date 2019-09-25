import tensorflow as tf
import numpy as np

class Models(object):
      
  def __init__(self, args):
      
      self.model_type = args.model_type         # model type - default is 'conv_skip'
      self.image_dim = args.image_dim           # image dimension [ROWS,COLS,CHANNELS]
      self.num_outputs = args.num_outputs       # number of outputs in segmentation = num_classes, in matting = 1, in dehazing = 3
      self.num_classes = args.num_classes       # number of classes, for matting: 2 (foreground+background), for Pascal VOC segmentation: 21
      self.seed = args.seed                     # random seed
      self.num_net_blocks = args.num_net_blocks # number of dilated residual blocks (see paper) 
      self.num_reg_conv = args.num_reg_conv     # number of regular conv layers in each block (not including the final dilated conv)
      self.use_softmax = args.use_softmax       # boolean indicating if there's an additional softmax in the end - for segmentation = True
      self.padding_type = args.padding_type     # 'None'/'REFLECT'
      self.app = args.app                       # application: 'seg'/'matte'/'dehaze' 
      self.f_size = [args.filter_size]*args.num_net_blocks          # vector the size of "rate" holding the filter size in each conv layer within the block
      self.num_filters = [args.num_filters]*args.num_net_blocks     # vector the size of "rate" holding the width of each conv layer within the block
      self.rate = [2**i for i in range(self.num_net_blocks)]        # vector of size of num dilated residual blocks holding each block's final layers' dilation rate
      self.pad_size = ((args.filter_size-1)/2)*(args.num_reg_conv*args.num_net_blocks+1) + ((args.filter_size-1)/2)*np.sum(self.rate)

  def define_weights(self, f_size, in_size, out_size, init_std):
    
      weights = tf.Variable(tf.truncated_normal([f_size[0],f_size[1],in_size,out_size],stddev = init_std, seed=self.seed),"weights")
      return weights

  def define_biases(self, out_size):
      
      biases = tf.Variable(tf.zeros(out_size),"biases")
      return biases

  def my_conv(self, x, rate, f_size, in_size, out_size, stride, with_relu, phase, name):
      
    with tf.name_scope(name):          
          weights = self.define_weights([f_size,f_size],in_size,out_size,0.1)
          biases = self.define_biases(out_size)
          if (rate == 0):
              if self.padding_type != 'None':
                  conv = tf.nn.conv2d(x,weights,strides=stride,padding='VALID',name='conv')
              else:
                  conv = tf.nn.conv2d(x,weights,strides=stride,padding='SAME',name='conv')
          else:              
              if self.padding_type != 'None':
                  conv = tf.nn.atrous_conv2d(x,weights,rate=rate,padding='VALID',name='conv')         
              else:
                  conv = tf.nn.atrous_conv2d(x,weights,rate=rate,padding='SAME',name='conv')         
          conv = tf.layers.batch_normalization(conv + biases, training=phase)
          if with_relu:
              conv = tf.nn.relu(conv)
    return conv

  def dilated_skip_block(self, x, app, rate, f_size, in_size, out_size, stride, phase, name):

    out = x      
    with tf.name_scope(name):
        for i in range(self.num_reg_conv):
            if (i == 0):
                out = self.my_conv(out, 0, f_size, in_size, out_size, stride, True, phase, 'conv_'+str(i+1))
            else:
                out = self.my_conv(out, 0, f_size, out_size, out_size, stride, True, phase, 'conv_'+str(i+1))        
        out = self.my_conv(out, rate, f_size, out_size,out_size, stride, False, phase,'dilated_conv')
        if self.padding_type != 'None':
            gap = ((f_size - 1)/2)*self.num_reg_conv + rate
            out = tf.add(out,x[:,gap:-gap,gap:-gap,:]) #fusing]
        else:
            out = tf.add(out, x) # adiition skip connection
        # if app is 'dehaze', there's no ReLU after the dilated residual block. If it's 'seg' or 'matte' there is
        if (app == 'seg') or (app == 'matte'):
            out = tf.nn.relu(out)
    return out

  def build_model(self, x, phase):
                
    if self.model_type == 'conv_skip':

        out = x
        if (self.padding_type != 'None'):
            out = tf.pad(out,[[0,0],[self.pad_size,self.pad_size],[self.pad_size,self.pad_size],[0,0]],self.padding_type)
        out = self.my_conv(out,0,self.f_size[0],self.num_classes + self.image_dim[2],self.num_filters[0],[1,1,1,1],True,phase,"first_conv")
        for i in range(len(self.rate)):
            out = self.dilated_skip_block(out,self.app,self.rate[i],self.f_size[i],self.num_filters[i-1],self.num_filters[i],[1,1,1,1],phase,"block_"+str(i+1))
        out = self.my_conv(out,0,1,self.num_filters[-1],self.num_outputs,[1,1,1,1],False,phase,"final_conv")
       
    if self.use_softmax == True:
        network_out = tf.nn.softmax(out, dim = -1, name = 'softmax')
    else:
        network_out = out
    self.network_out = network_out # add network output op for prediction/evaluation
    return network_out