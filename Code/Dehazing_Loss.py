import tensorflow as tf

class Dehazing_Loss(object):
      
  def __init__(self, args):
      
    self.image_dim = args.image_dim               # natural image dimension
    self.power = args.power                       # powers for smoothness and data terms in energy functions
    self.strength = args.strength                 # stregnth of smoothness and data terms in energy functions  
    self.dehaze_win = args.dehaze_win             # default window size for dehazing [15,15]
    self.omega = args.dehaze_omega                # default omega value in dehazing equation 0.95
    self.patch_size = args.matte_win              # default patch size is 3x3
    self.eps = args.dehaze_eps                    # epsilon parameter in energy function 10^-5
    self.t0 = args.dehaze_thresh                  # threshold in dehazing reconstruction 0.1
    self.pixels_in_patch = self.patch_size[0]*self.patch_size[1]
    self.num_patches = (self.image_dim[0]-(self.patch_size[0]-1))*(self.image_dim[1]-(self.patch_size[1]-1))

  def _loss(self, y_true, y_pred):
      
      # y_true contains the input hazy image, y_pred is the predicted transmssion of the network
      # y_pred is of size [NUM_BATCHES,NUM_ROWS,NUM_COLS,1]
      # t_true is of size [NUM_BATCHES,NUM_ROWS,NUM_COLS,NUM_CHANNELS]
      
      # compute initial coarse transmission map using hazing physical model
      with tf.name_scope('compute_coarse_transmission'):
          hazy_image = y_true
          patch_size = self.dehaze_win[0]*self.dehaze_win[1]
          paddings = tf.constant([[0,0],[(self.dehaze_win[0]-1)/2,(self.dehaze_win[0]-1)/2],[(self.dehaze_win[1]-1)/2,(self.dehaze_win[1]-1)/2],[0,0]])
          y_padded = tf.pad(hazy_image, paddings, "SYMMETRIC")
          num_patches = self.image_dim[0]*self.image_dim[1]
          y_patches = tf.extract_image_patches(y_padded, (1,self.dehaze_win[0],self.dehaze_win[1],1),(1,1,1,1),(1,1,1,1),'VALID',name='y_true_patches')
          initial_DCP = tf.reduce_min(y_patches,axis=-1)
          initial_DCP = tf.reshape(initial_DCP,(-1,num_patches))
          _, inds = tf.nn.top_k(initial_DCP,int(0.001*num_patches))

          hazy_image = tf.reshape(hazy_image,(-1,self.image_dim[0]*self.image_dim[1],self.image_dim[2]))
          
          B = tf.shape(hazy_image)[0]
          batch_inds = tf.range(B)
          batch_inds = tf.reshape(batch_inds, [-1, 1])    
          batch_inds = tf.tile(batch_inds, [1, tf.shape(inds)[1]])  
          batch_inds = tf.reshape(batch_inds, [-1]) 
          
          pixel_inds = tf.squeeze(tf.reshape(inds,[-1,1]))          
          inds_for_gather = tf.stack((batch_inds,pixel_inds),axis=-1)          
          pixels = tf.gather_nd(hazy_image,inds_for_gather)
          brightest_pixels = tf.reshape(pixels,(B,tf.shape(inds)[1],self.image_dim[2]))
          atmo_vec = tf.reduce_max(brightest_pixels,axis=1)
          
          y_patches = tf.reshape(y_patches,(B,self.image_dim[0],self.image_dim[1],patch_size,self.image_dim[2]))
          y_patches = tf.transpose(y_patches,(1,2,3,0,4))
          normalized_patches = tf.divide(y_patches,atmo_vec)
          normalized_patches = tf.transpose(normalized_patches,(3,0,1,2,4))
          normalized_patches = tf.reshape(normalized_patches,(B,self.image_dim[0],self.image_dim[1],patch_size*self.image_dim[2]))
          DCP = tf.reduce_min(normalized_patches,axis=-1)
          t_tilde = tf.add(-tf.multiply(DCP,self.omega),1)

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
          images = y_true[:,:,:,:] # C channels (C=1 for greyscale)
      
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
          matching_difference = tf.abs(tf.subtract(y_pred[:,:,:,0], t_tilde), name = 'seed_probs_diff')
          data_pow = tf.constant(self.power['data_term'],dtype=tf.float32,name='data_pow')
          data_term = tf.pow(matching_difference, data_pow,name='difference_pow')
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