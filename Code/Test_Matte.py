import numpy as np
import tensorflow as tf
import os
from os.path import dirname, abspath
from PIL import Image
from Models import Models
import train_utils as train_utils

parent_dir = dirname(dirname(abspath(__file__))) 
results_folder = parent_dir + '/Results/' + 'matte'
checkpoints_path = results_folder + '/checkpoints'
checkpoint = [59]
config_file_path = results_folder + '/params.ini'
args = train_utils.parse_params(config_file_path)  
args.image_dim = [128,128,3]
L = 2 # num classes = 2: foreground + background

which_trimap = 'trimaps_1' # 'trimaps_1'/'trimaps_2'
alpha_directory = parent_dir + '/Datasets/Matte/alphamatting/'
output_dir = results_folder + '/output_' + which_trimap
if not os.path.exists(output_dir + '/network'):
    os.makedirs(output_dir + '/network')

# create TF placeholders and construct network
tf.reset_default_graph()
in_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, args.image_dim[2]+L], name="in_placeholder")
out_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,1],name='out_placeholder')
phase = tf.placeholder(tf.bool,name='phase')
net_class = Models(args)
net_class.build_model(in_placeholder, phase)

# create TF session and upload saved patameters to network
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoints_path)
if ckpt and ckpt.model_checkpoint_path: 
    ckpt_path = checkpoints_path + '/my_model-' + str(checkpoint[0])
    saver.restore(sess, ckpt_path)
    
# perform inference over images in '/Datasets/Matte/alpha_matting'
images = sorted(os.listdir(alpha_directory+"/images"))
markers = sorted(os.listdir(alpha_directory+ "/" + which_trimap))
ground_truth = sorted(os.listdir(alpha_directory+"/ground_truth"))

for i in range(len(images)):
    
    print('image %d/%d' % (i+1,len(images)))
    image_name = images[i][:images[i].index(".")]   
    im = Image.open(alpha_directory+"/images"+"/"+images[i])
    im = np.array(im).astype(np.float32)/255
    mark = Image.open(alpha_directory + "/" + which_trimap + "/"+markers[i])
    mark = np.array(mark).astype(np.float32)
    curr_dim = np.shape(im)
    input_im = np.zeros((curr_dim[0],curr_dim[1],L + curr_dim[2]))
    input_im[:,:,L:] = im
    background_im = np.zeros((curr_dim[0],curr_dim[1]))  
    background_im[mark==0] = 1
    foreground_im = np.zeros((curr_dim[0],curr_dim[1]))
    foreground_im[mark==255] = 1
    input_im[:,:,0] = background_im
    input_im[:,:,1] = foreground_im
    ground_truth = Image.open(alpha_directory+"/ground_truth"+"/"+images[i])
    ground_truth = np.array(ground_truth).astype(np.float32)/255.0
    ground_truth = np.mean(ground_truth,axis=-1)
        
    input_im[:,:,L:] = input_im[:,:,L:] - args.rgb_avg
    im_for_tf = np.expand_dims(input_im,axis=0)
    feed_dict = {in_placeholder: im_for_tf, phase: False}
    alpha_network = np.squeeze(sess.run(net_class.network_out, feed_dict=feed_dict))
    alpha_network = np.minimum(np.maximum(alpha_network, 0), 1)
    Image.fromarray((alpha_network*65535).astype(np.uint16)).save(output_dir + "/network/" + image_name + '.tif')
    
sess.close()