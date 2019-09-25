import numpy as np
import tensorflow as tf
import os
from os.path import dirname, abspath
from PIL import Image
from Models import Models
import train_utils as train_utils
import Utils as utils

parent_dir = dirname(dirname(abspath(__file__))) 
results_folder = parent_dir + '/Results/' + 'dehaze'
checkpoints_path = results_folder + '/checkpoints/'
image_folder = parent_dir + '/Datasets/Dehaze/Qual/hazy'
config_file_path = results_folder + '/params.ini'
args = train_utils.parse_params(config_file_path)  
args.image_dim = [128,128,3]
checkpoint = 30
 
# create TF placeholders and build network
tf.reset_default_graph()
in_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, args.image_dim[2]], name="in_placeholder")
out_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,1],name='out_placeholder')
phase = tf.placeholder(tf.bool,name='phase')
net_class = Models(args)
net_class.build_model(in_placeholder,phase)

# open TF session and load saved model
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoints_path)
if ckpt and ckpt.model_checkpoint_path: 
    ckpt_path = checkpoints_path + 'my_model-' + str(checkpoint)
    saver.restore(sess, ckpt_path)
        
# create output folder
output_folder = results_folder + '/Qual_output/' + str(checkpoint)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
            
# iterate over images in '/Datasets/Dehaze/Qual'
image_list = sorted(os.listdir(image_folder))
    
for i in range(len(image_list)):
    
    file_name = image_list[i]
    image = np.array(Image.open(image_folder + '/' + file_name))
    print ('image (%d/%d)' % (i+1,len(image_list)))
    im_for_tf = np.expand_dims(image/255.0,axis=0)
    t_network = np.squeeze(sess.run([net_class.network_out], feed_dict={in_placeholder: im_for_tf, phase:False}))
    J_network = utils.clean_from_hazy(image,t_network,args.dehaze_win,args.dehaze_omega,args.dehaze_thresh)
    J_network = (J_network - np.min(J_network))/(np.max(J_network)-np.min(J_network))
    Image.fromarray(utils.prep_save(J_network)).save(output_folder + "/" + file_name)
            
sess.close()