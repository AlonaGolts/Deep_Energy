import tensorflow as tf

class Segmentation_Loss(object):

  def __init__(self, args):
      
    self.image_dim = args.image_dim               # natural image dimension
    self.num_classes = args.num_classes           # number of classes - for pascal VOC 2012: 21
    self.beta = args.beta                         # global parameter for random walker 
    self.num_neighbours = args.num_neighbors      # number of neighbours, default is 4 for 4-connectivity 
    self.power = args.power                       # powers for smoothness and data terms in energy functions
    self.strength = args.strength                 # stregnth of smoothness and data terms in energy functions

  def _loss(self, y_true, y_pred):
      
      # y_true contains the input image and seeds, y_pred is the prediction of the network
      # y_pred is of size [NUM_BATCHES,NUM_ROWS,NUM_COLS,NUM_CLASSES]
      # y_true is of size [NUM_BATCHES,NUM_ROWS,NUM_COLS,NUM_CLASSES+NUM_CHANNELS]

      # Smoothness term: compute y_pred_expand
      with tf.name_scope('compute_y_pred_expand'):
          y_pred_exp = tf.multiply(tf.expand_dims(y_pred, axis=-1), tf.ones([self.num_neighbours]),name = 'y_pred_expand')
     
      # separate input data to natural image and seeds
      with tf.name_scope('separate_image_and_seeds'):
          images = y_true[:,:,:,self.num_classes:] # C channels (C=1 for greyscale)
          seeds = y_true[:,:,:,:self.num_classes] # K classes 
          
      # Smoothness term: compute weights_neighbours.
      with tf.name_scope('compute_weights'):            
          image_expand = tf.multiply(tf.expand_dims(images,axis=-1),tf.ones([self.num_neighbours]),name='images_expand')      
          top_i = tf.expand_dims(tf.concat((tf.expand_dims(images[:,-1,:,:],axis=1),images[:,0:-1,:,:]),axis=1,name='top_n'),axis=-1)
          bottom_i = tf.expand_dims(tf.concat((images[:,1:,:,:],tf.expand_dims(images[:,0,:,:],axis=1)),axis=1,name='bottom_n'),axis=-1)
          left_i = tf.expand_dims(tf.concat((tf.expand_dims(images[:,:,-1,:],axis=2),images[:,:,0:-1,:]),axis=2,name='left_n'),axis=-1)
          right_i = tf.expand_dims(tf.concat((images[:,:,1:,:],tf.expand_dims(images[:,:,0,:],axis=2)),axis=2,name='right_n'),axis=-1)
          image_neighbours = tf.concat((top_i,bottom_i,left_i,right_i),axis=-1,name='concat_neighbours')
          weights_power = tf.reduce_sum(tf.pow((image_expand - image_neighbours),2),axis=3,name='weights_distance') # sum for the ||g_i-g_j||^2
          weights_neighbours = tf.expand_dims(tf.exp(tf.multiply(weights_power,-self.beta)),axis=3,name='weights_power')
          weights_neighbours = tf.multiply(weights_neighbours,tf.ones((self.num_classes,self.num_neighbours)),name='weights_final') # copy the weight value to all classes
          
      # Smoothness term: compute y_neighbours.
      with tf.name_scope('compute_y_neighbours'):
          top_n = tf.expand_dims(tf.concat((tf.expand_dims(y_pred[:,-1,:,:],axis=1),y_pred[:,0:-1,:,:]),axis=1,name='top_n'),axis=-1)
          bottom_n = tf.expand_dims(tf.concat((y_pred[:,1:,:,:],tf.expand_dims(y_pred[:,0,:,:],axis=1)),axis=1,name='bottom_n'),axis=-1)
          left_n = tf.expand_dims(tf.concat((tf.expand_dims(y_pred[:,:,-1,:],axis=2),y_pred[:,:,0:-1,:]),axis=2,name='left_n'),axis=-1)
          right_n = tf.expand_dims(tf.concat((y_pred[:,:,1:,:],tf.expand_dims(y_pred[:,:,0,:],axis=2)),axis=2,name='right_n'),axis=-1)
          y_neighbours = tf.concat((top_n,bottom_n,left_n,right_n),axis=-1,name='concat_neighbours')
          
      # Data term.
      with tf.name_scope('data_term'):
          matching_indicator = tf.reduce_sum(seeds, axis=3, name ='seed_locs')  # sum over k
          matching_difference = tf.abs(tf.subtract(y_pred, seeds), name = 'seed_probs_diff')
          data_pow = tf.constant(self.power['data_term'],dtype=tf.float32,name='data_pow')
          matching_penalty = tf.reduce_sum(tf.pow(matching_difference, data_pow),axis=3, name = 'penalty')  # sum over k
          data_term = tf.multiply(matching_indicator, matching_penalty,name='final_data_term')
          data_term = tf.reduce_sum(data_term, axis=2, name='sum_cols')  # sum over j
          data_term = tf.reduce_sum(data_term, axis=1, name='sum_rows')  # sum over i

      # Smoothness term: put it all together.
      with tf.name_scope('smoothness_term'):
          neighbour_difference = tf.abs(tf.subtract(y_pred_exp, y_neighbours),name='neighbour_difference')
          smooth_pow = tf.constant(self.power['smoothness_term'], dtype=tf.float32, name='smooth_pow')
          neighbour_difference_power = tf.pow(neighbour_difference,smooth_pow,name='neighbour_difference_power')
          smoothness_term = tf.multiply(neighbour_difference_power,weights_neighbours,name='final_smooth_term')
          smoothness_term = tf.reduce_sum(smoothness_term, axis=4, name='sum_neighbours')
          smoothness_term = tf.reduce_sum(smoothness_term, axis=3, name='sum_classes')
          smoothness_term = tf.reduce_sum(smoothness_term, axis=2, name='sum_cols')
          smoothness_term = tf.reduce_sum(smoothness_term, axis=1, name='sum_rows')

      # List of terms.
      with tf.name_scope('sum_terms'):
          s_strength = tf.constant(0.5*self.strength['smoothness_term'],dtype=tf.float32,name='s_strength')
          d_strength = tf.constant(self.strength['data_term'],dtype=tf.float32,name='d_strength')
          terms = []
          only_smoothness = tf.multiply(smoothness_term, s_strength,name='s_term')
          only_data = tf.multiply(data_term, d_strength,name='d_term')
          terms.append(only_smoothness)
          terms.append(only_data)
          self.loss_smooth = tf.reduce_sum(only_smoothness,name='only_smooth_loss')
          self.loss_data = tf.reduce_sum(only_data,name='only_data_loss')

          # Add up the terms.
          loss_per_point = tf.add_n(terms, name='add_terms')
          final_loss = tf.reduce_sum(loss_per_point, name='final_loss')
      
      return final_loss