import numpy as np
import tensorflow as tf
import os
from os.path import dirname, abspath
from PIL import Image
import csv
from Models import Models
from Matting_Loss import Matting_Loss
import Utils as utils
import train_utils as train_utils

#%%
parent_dir = dirname(dirname(abspath(__file__))) 
results_folder = parent_dir + '/Results/' + 'matte'
checkpoints_path = results_folder + '/checkpoints'
checkpoint = [59]

config_file_path = results_folder + '/params.ini'
args = train_utils.parse_params(config_file_path)  
args.image_dim = [128,128,3]
L = 2 # num classes = 2: foreground + background

compute_analytic = False
compute_loss = False

which_trimap = 'trimaps_1' # 'trimaps_1'/'trimaps_2'
alpha_directory = parent_dir + '/Datasets/Matte/alphamatting/'
output_dir = results_folder + '/output_' + which_trimap
numeric_rep_path = results_folder + "/results_" + which_trimap + ".csv"

#%% load saved network parameters and open new session
tf.reset_default_graph()
in_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, args.image_dim[2]+L], name="in_placeholder")
out_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,1],name='out_placeholder')
phase = tf.placeholder(tf.bool,name='phase')
net_class = Models(args)
net_class.build_model(in_placeholder, phase)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoints_path)
if ckpt and ckpt.model_checkpoint_path: 
    ckpt_path = checkpoints_path + '/my_model-' + str(checkpoint[0])
    saver.restore(sess, ckpt_path)
    
#%% Compute statistics of MSE, SAD and runtime on the full-sized images of alphamatting.com
if not os.path.exists(output_dir + '/network'):
    os.makedirs(output_dir + '/network')
if compute_analytic:
    if not os.path.exists(output_dir + '/analytic'):
        os.makedirs(output_dir + '/analytic')
image_list = os.listdir(alpha_directory+"/images")
marker_list = os.listdir(alpha_directory+ "/" + which_trimap)
ground_truth = os.listdir(alpha_directory+"/ground_truth")

res_file  = open(numeric_rep_path, "wb")
writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
if compute_analytic:
    header_row = 'image','analytic_loss','network_loss','analytic_MSE','network_MSE','analytic_SAD', 'network_SAD', 'analytic_time','network_time'
else:
    header_row = 'image','network_loss','network_MSE','network_SAD', 'network_time'
writer.writerow(header_row)

images = sorted(image_list)
markers = sorted(marker_list)
ground_truth = sorted(ground_truth)

MSE_analytic = np.zeros(len(image_list))
MSE_network = np.zeros(len(image_list))
SAD_analytic = np.zeros(len(image_list))
SAD_network = np.zeros(len(image_list))
loss_analytic = np.zeros(len(image_list))
loss_network = np.zeros(len(image_list))
time_analytic = np.zeros(len(image_list))
time_network = np.zeros(len(image_list))

for i in range(len(images)):
    
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
    
    if compute_analytic == True:
        utils.tic()
        alpha_analytic, loss_analytic[i], _, _ = utils.analytic_matte(input_im,L,args.strength, args.matte_eps, args.matte_win_size)
        time_analytic[i] = utils.toc()
        alpha_analytic = np.minimum(np.maximum(alpha_analytic, 0), 1)
        MSE_analytic[i] = utils.compute_MSE_with_trimap(alpha_analytic,ground_truth,mark)
        SAD_analytic[i] = utils.compute_SAD_with_trimap(alpha_analytic,ground_truth,mark)
        Image.fromarray((alpha_analytic*65535).astype(np.uint16)).save(output_dir + "/analytic/" + image_name + '.tif')
        
    input_im[:,:,L:] = input_im[:,:,L:] - args.rgb_avg
    im_for_tf = np.expand_dims(input_im,axis=0)
    feed_dict = {in_placeholder: im_for_tf, phase: False}
    utils.tic()
    alpha_network = sess.run(net_class.network_out, feed_dict=feed_dict)
    time_network[i] = utils.toc()
    
    if compute_loss:
        args.image_dim = curr_dim
        matte_loss = Matting_Loss(args)
        loss_op = matte_loss._loss(in_placeholder,out_placeholder)
        feed_dict = {in_placeholder: im_for_tf, out_placeholder: alpha_network}
        loss_network[i] = sess.run([loss_op],feed_dict=feed_dict)[0]
    
    alpha_network = np.squeeze(alpha_network)
    alpha_network = np.minimum(np.maximum(alpha_network, 0), 1)
    MSE_network[i] = utils.compute_MSE_with_trimap(alpha_network,ground_truth,mark)
    SAD_network[i] = utils.compute_SAD_with_trimap(alpha_network,ground_truth,mark)

    Image.fromarray((alpha_network*65535).astype(np.uint16)).save(output_dir + "/network/" + image_name + '.tif')
    
    if compute_analytic:
        print('%s, analytic_loss: %.2f, network_loss: %.2f, analytic_MSE: %.5f, network_MSE: %.5f, analytic_SAD: %.2f, network_SAD: %.2f, analytic_time: %.2f, network_time: %.2f' % \
              (image_name + '.png',loss_analytic[i],loss_network[i],MSE_analytic[i],MSE_network[i],SAD_analytic[i],SAD_network[i],time_analytic[i],time_network[i]))
    else:
        print('%s, network_loss: %.2f, network_MSE: %.5f, network_SAD: %.2f, network_time: %.2f' % \
              (image_name + '.png',loss_network[i],MSE_network[i],SAD_network[i],time_network[i]))
        
    if compute_analytic:
        row = image_name,'%.2f' % loss_analytic[i], '%.2f' % loss_network[i], '%.5f'% MSE_analytic[i], '%.5f' % MSE_network[i], \
            '%.2f' % SAD_analytic[i], '%.2f' % SAD_network[i], '%.2f' % time_analytic[i], '%.2f' % time_network[i]
    else:
        row = image_name,'%.2f' % loss_network[i], '%.5f' % MSE_network[i], \
            '%.2f' % SAD_network[i], '%.2f' % time_network[i]
    writer.writerow(row)
    res_file.flush()

if compute_analytic:
    row = 'mean','%.2f' % np.mean(loss_analytic), '%.2f' % np.mean(loss_network), '%.3f' % np.mean(MSE_analytic), '%.3f' % np.mean(MSE_network), \
         '%.2f' % np.mean(SAD_analytic), '%.2f' % np.mean(SAD_network), '%.2f' % np.mean(time_analytic), '%.2f' % np.mean(time_network)
else:
    row = 'mean', '%.2f' % np.mean(loss_network), '%.3f' % np.mean(MSE_network), \
         '%.2f' % np.mean(SAD_network), '%.2f' % np.mean(time_network)
writer.writerow(row)
res_file.flush()
res_file.close()
sess.close()