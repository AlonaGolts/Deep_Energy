import numpy as np
import tensorflow as tf
import os
from os.path import dirname, abspath
from PIL import Image
from Models import Models
from Segmentation_Loss import Segmentation_Loss
import train_utils as train_utils
import Utils as utils

#%% Params
parent_dir = dirname(dirname(abspath(__file__))) 
results_folder = parent_dir + '/Results/seg_beta_1000'
checkpoints_path = results_folder + '/checkpoints/'
output_path = results_folder + '/output_pascal'
checkpoint = [25]

pascal_path = parent_dir + '/Datasets/Seg/VOC2012_val'
config_file_path = results_folder + '/params.ini'
args = train_utils.parse_params(config_file_path)  
L = args.num_classes
args.image_dim = [128,128,3]
seg_loss = Segmentation_Loss(args)
seed_color = 22
compute_analytic = False
compute_loss = True
save_images = True

#%% load saved network parameters and open new session
tf.reset_default_graph()
in_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, L+args.image_dim[2]], name="in_placeholder")
out_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,L],name='out_placeholder')
phase = tf.placeholder(tf.bool,name='phase')
net_class = Models(args)
net_class.build_model(in_placeholder, phase)
palette = Image.open(parent_dir + '/Datasets/Seg/VOC2012_val/palette/2007_000039.png').getpalette()

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoints_path)
if ckpt and ckpt.model_checkpoint_path: 
    ckpt_path = checkpoints_path + 'my_model-' + str(checkpoint[0])
    saver.restore(sess, ckpt_path)
    
#%% 
if save_images:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
image_list = sorted(os.listdir(pascal_path + '/images'))        
analytic_loss = np.zeros(len(image_list))
network_loss = np.zeros(len(image_list))
cc_network_seg = np.zeros(L**2)
cc_analytic_seg = np.zeros(L**2)
time_analytic_sol = np.zeros(len(image_list))
time_analytic_seg = np.zeros(len(image_list))
time_network_sol = np.zeros(len(image_list))
time_network_seg = np.zeros(len(image_list))

for i in range(len(image_list)):

    image_name = image_list[i][:image_list[i].index(".")] 
    orig_image = Image.open(pascal_path + '/images/' + image_name + '.jpg')
    orig_seed = Image.open(pascal_path + '/seeds/' + image_name + '.png')
    orig_gt = Image.open(pascal_path + '/gts/' + image_name + '.png')
    orig_dim = np.shape(orig_image)
        
    if save_images:
        orig_image.save(output_path + "/" + image_name + "_1.bmp")
        gt_with_markers = np.copy(orig_gt)
        gt_with_markers[np.array(orig_seed) > 0] = seed_color
        gt_with_markers = Image.fromarray(gt_with_markers.astype(np.uint8))
        utils.save_seg(save_images,gt_with_markers,palette,output_path + "/" + image_name + "_4.bmp")
        
    image = np.array(orig_image.resize((args.image_dim[1],args.image_dim[0]),resample=Image.BILINEAR))/255.0
    seed = np.array(orig_seed.resize((args.image_dim[1],args.image_dim[0]),resample=Image.NEAREST))
    gt = np.array(orig_gt.resize((args.image_dim[1],args.image_dim[0]),resample=Image.NEAREST))
    image_and_seeds = utils.concat_image_seeds(image,orig_seed,seed,L)
        
    if compute_analytic == True:
        utils.tic()
        analytic_sol,analytic_loss[i],_,_ = utils.analytic_seg(image_and_seeds,L,args.strength,args.beta)
        time_analytic_sol[i] = utils.toc()
        utils.tic()
        analytic_seg = np.argmax(analytic_sol,axis=-1)
        analytic_seg = Image.fromarray(analytic_seg.astype(np.uint8)).resize((orig_dim[1],orig_dim[0]),resample=Image.NEAREST)
        time_analytic_seg[i] = utils.toc()
        analytic_back = Image.fromarray(utils.prep_save(analytic_sol[:,:,0])).resize((orig_dim[1],orig_dim[0]),resample=Image.BILINEAR)
        if save_images:
            analytic_back.save(output_path + "/" + image_name + "_2.bmp") 
        utils.save_seg(save_images,analytic_seg,palette,output_path + "/" + image_name + "_5.bmp")
        cc_analytic_seg += utils.conf_counts_add(analytic_seg,orig_gt,L)
                
    image_and_seeds[:,:,L:] = image_and_seeds[:,:,L:] - args.rgb_avg
    image_and_seeds = np.expand_dims(image_and_seeds, axis=0)
    utils.tic()
    network_sol = sess.run(net_class.network_out, feed_dict={in_placeholder: image_and_seeds, phase: False})
    time_network_sol[i] = utils.toc()
    if compute_loss:
        loss_op = seg_loss._loss(in_placeholder,out_placeholder)
        feed_dict = {in_placeholder: image_and_seeds, out_placeholder: network_sol}
        network_loss[i] = sess.run(loss_op,feed_dict=feed_dict)
    network_sol = np.squeeze(network_sol)
        
    utils.tic()
    network_seg = np.argmax(network_sol,axis=-1)
    network_seg = Image.fromarray(network_seg.astype(np.uint8)).resize((orig_dim[1],orig_dim[0]),resample=Image.NEAREST)
    time_network_seg[i] = utils.toc()
    network_back = Image.fromarray(utils.prep_save(network_sol[:,:,0])).resize((orig_dim[1],orig_dim[0]),resample=Image.BILINEAR)
    if save_images:
        network_back.save(output_path + "/" + image_name + "_3.bmp")
    utils.save_seg(save_images,network_seg,palette,output_path + "/" + image_name + "_6.bmp")
    cc_network_seg += utils.conf_counts_add(network_seg,orig_gt,L)
        
    if compute_analytic:
        print ('Image %d/%d: analytic_loss: %.5f, network_loss: %.5f, time_a_sol: %.3f, time_n_sol: %.3f, speedup: %.3f' \
           % (i+1,len(image_list),analytic_loss[i],network_loss[i], time_analytic_sol[i],time_network_sol[i], time_analytic_sol[i]/time_network_sol[i]))
    else:
        print ('Image %d/%d: network_loss: %.5f, time_n_sol: %.3f' % (i+1,len(image_list),network_loss[i], time_network_sol[i]))

# note that mIOU can be computed accurately only for the entire 'val' set of Pascal VOC 2012 
acc_network_seg = utils.conf_mat(cc_network_seg,L)
acc_analytic_seg = utils.conf_mat(cc_analytic_seg,L) if compute_analytic else 0
if compute_analytic:
    print ('analytic_mIOU_seg: %.5f, network_mIOU_seg: %.5f, analytic_loss: %.5f, network_loss: %.5f' % \
       (np.mean(acc_analytic_seg),np.mean(acc_network_seg),np.mean(analytic_loss),np.mean(network_loss)))
else:
    print ('network_mIOU_seg: %.5f, network_loss: %.5f' % \
       (np.mean(acc_network_seg),np.mean(network_loss)))
    
sess.close()
