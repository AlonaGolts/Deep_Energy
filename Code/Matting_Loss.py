import tensorflow as tf

class Matting_Loss(object):

  def __init__(self, args):
      
    self.image_dim = args.image_dim              # natural image dimension
    self.num_classes = args.num_classes          # number of classes: default is 2: background and foreground
    self.power = args.power                      # powers for smoothness and data terms in energy functions
    self.strength = args.strength                # stregnth of smoothness and data terms in energy functions  
    self.patch_size = args.matte_win             # default patch size is 3x3
    self.eps = args.matte_eps                    # epsilon parameter in energy function
    self.pixels_in_patch = self.patch_size[0]*self.patch_size[1]
    self.num_patches = (self.image_dim[0]-(self.patch_size[0]-1))*(self.image_dim[1]-(self.patch_size[1]-1))

  def _loss(self, y_true, y_pred):
      
      # y_true contains the input image and seeds, y_pred is the predicted alpha of the network
      # y_pred is of size [NUM_BATCHES,NUM_ROWS,NUM_COLS,1]
      # t_true is of size [NUM_BATCHES,NUM_ROWS,NUM_COLS,NUM_CLASSES+NUM_CHANNELS]

      # Smoothness term: compute y.
      with tf.name_scope('expand_y_pred'):
          y = tf.extract_image_patches(y_pred, (1,self.patch_size[0],self.patch_size[1],1),(1,1,1,1),(1,1,1,1),'VALID',name='y_pred_patches')
          y = tf.reshape(y,(-1,self.num_patches,self.pixels_in_patch),name='y_pred_patces_reshape')
          y_I = tf.tile(y,(1,1,self.pixels_in_patch),name='y_pred_tiled')
          # start of hack to recreate np.repeat which is not implemented in this version of tf
          temp = tf.transpose(y,(2,0,1))
          idx = tf.range(self.pixels_in_patch)
          idx = tf.reshape(idx, [-1, 1])    
          idx = tf.tile(idx, [1, self.pixels_in_patch])  
          idx = tf.reshape(idx, [-1])       
          y_J = tf.gather(temp,idx)
          y_J = tf.transpose(y_J,(1,2,0),name='y_pred_repeat')
          # end of tf.repeat
      
      # Separate between image data and seed data
      with tf.name_scope('separate_image_and_seed_data'):
          images = y_true[:,:,:,self.num_classes:] # C channels (C=1 for greyscale)
          seeds = y_true[:,:,:,:self.num_classes] # 2 classes - foreground and background
      
      # Smoothness term: compute weights_neighbours.
      with tf.name_scope('compute_weights'):  
          r_patches = tf.extract_image_patches(tf.expand_dims(images[:,:,:,0],axis=-1),\
                (1,self.patch_size[0],self.patch_size[1],1),(1,1,1,1),(1,1,1,1),'VALID',name='R_patches')
          g_patches = tf.extract_image_patches(tf.expand_dims(images[:,:,:,1],axis=-1),\
                (1,self.patch_size[0],self.patch_size[1],1),(1,1,1,1),(1,1,1,1),'VALID',name='G_patches')
          b_patches = tf.extract_image_patches(tf.expand_dims(images[:,:,:,2],axis=-1),\
                (1,self.patch_size[0],self.patch_size[1],1),(1,1,1,1),(1,1,1,1),'VALID',name='B_patches')
          im_patches = tf.concat((tf.expand_dims(r_patches,axis=-1),tf.expand_dims(g_patches,axis=-1),\
                                  tf.expand_dims(b_patches,axis=-1)),axis=-1,name='all_image_patches')
          im_patches = tf.reshape(im_patches,(-1,self.num_patches,self.pixels_in_patch,self.image_dim[2]),name='image_patches')
          mean_patches = tf.reduce_mean(im_patches, keep_dims = True, axis=2,name='mean_image_patches')
          XX_T = tf.divide(tf.matmul(tf.transpose(im_patches,(0,1,3,2)),im_patches),float(self.pixels_in_patch),name='image_patches_squared')
          UU_T = tf.matmul(tf.transpose(mean_patches,(0,1,3,2)),mean_patches,name='mean_patches_squared')
          var_patches = tf.subtract(XX_T,UU_T,name='var_image_patches')
          matrix_to_invert = tf.add(tf.multiply(tf.divide(self.eps,self.pixels_in_patch),tf.eye(self.image_dim[2])),var_patches,name='matrix_to_invert')
          var_fac = tf.matrix_inverse(matrix_to_invert,name='inverted_matrix')
          weights = tf.matmul(im_patches-mean_patches,var_fac,name='weights')
          weights = tf.add(tf.matmul(weights,tf.transpose(im_patches-mean_patches,(0,1,3,2))),1,name='weights_plus_1')
          weights = tf.multiply(weights,tf.divide(1,self.pixels_in_patch),name='weights_divided')
          weights = tf.reshape(weights,(-1,self.num_patches,(self.pixels_in_patch*self.pixels_in_patch)),name='final_weights')
          
      # Smoothness term: put it all together.
      with tf.name_scope('smooth_term'):
          neighbour_difference = tf.abs(tf.subtract(y_I, y_J),name='neighbour_difference')
          smooth_pow = tf.constant(self.power['smoothness_term'], dtype=tf.float32, name='smooth_pow')
          neighbour_difference_power = tf.pow(neighbour_difference,smooth_pow,name='neighbour_difference_pow')
          smoothness_term = tf.multiply(neighbour_difference_power,weights,name='smooth_term_final')
          smoothness_term = tf.reduce_sum(smoothness_term, axis=2, name='sum_cols')
          smoothness_term = tf.reduce_sum(smoothness_term, axis=1, name='sum_rows')
      
      # Data term.
      with tf.name_scope('data_term'):
          matching_indicator = tf.reduce_sum(seeds, axis=3, name = 'seed_locs')  
          a_true = seeds[:,:,:,1] # the "true" alpha map is simply an all-zeros image with the foreground pixels marked as '1'
          matching_difference = tf.abs(tf.subtract(y_pred[:,:,:,0], a_true), name = 'seed_probs_diff')
          data_pow = tf.constant(self.power['data_term'],dtype=tf.float32,name='data_pow')
          matching_penalty = tf.pow(matching_difference, data_pow,name='difference_pow')
          data_term = tf.multiply(matching_indicator, matching_penalty,name='data_term_final')
          data_term = tf.reduce_sum(data_term, axis=2, name='sum_cols')  # sum over j
          data_term = tf.reduce_sum(data_term, axis=1, name='sum_rows')  # sum over i
    
      # List of term.
      with tf.name_scope('sum_terms'):
          s_strength = tf.constant(0.5*self.strength['smoothness_term'],dtype=tf.float32,name='s_strength')
          d_strength = tf.constant(self.strength['data_term'],dtype=tf.float32,name='d_strength')
          terms = []
          only_smoothness = tf.multiply(smoothness_term, s_strength,name='s_term')
          only_data = tf.multiply(data_term, d_strength,name='d_term')
          terms.append(only_smoothness)
          terms.append(only_data)
          self.loss_smooth = tf.reduce_sum(only_smoothness,name='only_matte_loss')
          self.loss_data = tf.reduce_sum(only_data,name='only_data_loss')

          # Add up the terms.
          loss_per_point = tf.add_n(terms, name='add_terms')
          final_loss = tf.reduce_sum(loss_per_point, name='fina_loss')
      
      return final_loss