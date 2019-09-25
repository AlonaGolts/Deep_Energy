import numpy as np
import tensorflow as tf
import os
from os.path import dirname, abspath
from PIL import Image
from Models import Models
import train_utils as train_utils
import Utils as utils

#%% Params
parent_dir = dirname(dirname(abspath(__file__))) 
results_folder = parent_dir + '/Results/seg_beta_1000'
pascal_path = parent_dir + '/Datasets/Seg/VOC2012_val'
checkpoints_path = results_folder + '/checkpoints/'
output_path = results_folder + '/output_pascal'
checkpoint = [25]
config_file_path = results_folder + '/params.ini'
args = train_utils.parse_params(config_file_path)  
L = args.num_classes
args.image_dim = [128,128,3]

#%% load saved network parameters and open new session
tf.reset_default_graph()
in_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, L+args.image_dim[2]], name="in_placeholder")
out_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,L],name='out_placeholder')
phase = tf.placeholder(tf.bool,name='phase')
net_class = Models(args)
net_class.build_model(in_placeholder, phase)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoints_path)
if ckpt and ckpt.model_checkpoint_path: 
    ckpt_path = checkpoints_path + 'my_model-' + str(checkpoint[0])
    saver.restore(sess, ckpt_path)
    
#%% 
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
image_list = sorted(os.listdir(pascal_path + '/images'))        

for i in range(len(image_list)):

    print ('Image %d/%d' % (i+1,len(image_list)))
    image_name = image_list[i][:image_list[i].index(".")] 
    orig_image = Image.open(pascal_path + '/images/' + image_name + '.jpg')
    orig_seed = Image.open(pascal_path + '/seeds/' + image_name + '.png')
    orig_gt = Image.open(pascal_path + '/gts/' + image_name + '.png')
    orig_dim = np.shape(orig_image)
        
    orig_image.save(output_path + "/" + image_name + "_1.bmp")
    gt_with_markers = np.copy(orig_gt)
    gt_with_markers[np.array(orig_seed) > 0] = 22
    gt_with_markers = Image.fromarray(gt_with_markers.astype(np.uint8))
    palette = Image.open(parent_dir + '/Datasets/Seg/VOC2012_val/palette/2007_000039.png').getpalette()
    utils.save_seg(True,gt_with_markers,palette,output_path + "/" + image_name + "_4.bmp")
        
    image = np.array(orig_image.resize((args.image_dim[1],args.image_dim[0]),resample=Image.BILINEAR))/255.0
    seed = np.array(orig_seed.resize((args.image_dim[1],args.image_dim[0]),resample=Image.NEAREST))
    gt = np.array(orig_gt.resize((args.image_dim[1],args.image_dim[0]),resample=Image.NEAREST))
    image_and_seeds = utils.concat_image_seeds(image,orig_seed,seed,L)
                
    image_and_seeds[:,:,L:] = image_and_seeds[:,:,L:] - args.rgb_avg
    image_and_seeds = np.expand_dims(image_and_seeds, axis=0)
    network_sol = np.squeeze(sess.run(net_class.network_out, feed_dict={in_placeholder: image_and_seeds, phase: False}))
        
    network_seg = np.argmax(network_sol,axis=-1)
    network_seg = Image.fromarray(network_seg.astype(np.uint8)).resize((orig_dim[1],orig_dim[0]),resample=Image.NEAREST)
    network_back = Image.fromarray(utils.prep_save(network_sol[:,:,0])).resize((orig_dim[1],orig_dim[0]),resample=Image.BILINEAR)
    network_back.save(output_path + "/" + image_name + "_3.bmp")
    utils.save_seg(True,network_seg,palette,output_path + "/" + image_name + "_6.bmp")
    
sess.close()