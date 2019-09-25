import numpy as np
import tensorflow as tf
import os
from os.path import dirname, abspath
from PIL import Image
from Models import Models
from Dehazing_Loss import Dehazing_Loss
import train_utils as train_utils
import Utils as utils
import math
import csv
from skimage.measure import compare_ssim as ssim

#%% Params
parent_dir = dirname(dirname(abspath(__file__))) 
results_folder = parent_dir + '/Results/' + 'dehaze'
checkpoints_path = results_folder + '/checkpoints/'

config_file_path = results_folder + '/params.ini'
args = train_utils.parse_params(config_file_path)  
args.image_dim = [128,128,3]

compute_analytic = False
compute_SSIM = True
compute_loss = False
run_CPU = False

dataset_folder = parent_dir + '/Datasets/Dehaze/'
qual_path = dataset_folder + 'Qual/hazy'

compute_qual = True
# Note that Middlebury consumes a lot of memory. When choosing: 'Middlebury': True, use compute_analytic = False, compute_loss = False
compute_datasets = {'HSTS': True, 'SOTS_indoor': True, 'SOTS_outdoor': True, 'Middlebury': True}
save_images_dict = {'HSTS': True, 'SOTS_indoor': True, 'SOTS_outdoor': True, 'Middlebury': True}
checkpoints_dict = {'HSTS': [27], 'SOTS_indoor': [30], 'SOTS_outdoor': [27], 'Middlebury': [33]}
qual_checkpoints = [30]

data_folder_dict = {'HSTS': 'RESIDE_HSTS', 'SOTS_indoor': 'RESIDE_SOTS_indoor', \
                    'SOTS_outdoor': 'RESIDE_SOTS_outdoor', 'Middlebury': 'Middlebury'}
hazy_file_type_dict = {'HSTS': '.jpg', 'SOTS_indoor': '.png', 'SOTS_outdoor': '.jpg', 'Middlebury': '.bmp'}
gt_file_type_dict = {'HSTS': '.jpg', 'SOTS_indoor': '.png', 'SOTS_outdoor': '.png', 'Middlebury': '.png'}

#%% 
if compute_analytic:
    header_row = 'epoch', 'analytic_PSNR', 'network_PSNR', 'analytic_SSIM', 'network_SSIM', 'analytic_loss', 'network_loss', 'analytic_time', 'network_time'
else:
    header_row = 'epoch', 'network_PSNR', 'network_SSIM', 'network_loss', 'network_time'
    
for dataset in compute_datasets:
    
    if not(compute_datasets[dataset]):
        continue
    
    print ('\n################### Now computing %s ##################\n' % dataset)
               
    log_path = results_folder + "/" + dataset + "_results.csv"
    data_file = open(log_path, "wb")
    writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(header_row)
    
    data_folder = dataset_folder + data_folder_dict[dataset] 
    orig_path = data_folder + '/gt'
    hazy_path = data_folder + '/hazy'
    hazy_file_type = hazy_file_type_dict[dataset]
    gt_file_type = gt_file_type_dict[dataset]
    save_images = save_images_dict[dataset]
    
    image_list = sorted(os.listdir(hazy_path))
    PSNR_analytic = np.zeros(len(image_list))
    PSNR_network = np.zeros(len(image_list))
    loss_analytic = np.zeros(len(image_list))
    loss_network = np.zeros(len(image_list))
    SSIM_analytic = np.zeros(len(image_list))
    SSIM_network = np.zeros(len(image_list))
    time_analytic = np.zeros(len(image_list))
    time_network = np.zeros(len(image_list))   
    
    for checkpoint in checkpoints_dict[dataset]:
        
        # load saved network parameters and open new session
        tf.reset_default_graph()
        in_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, args.image_dim[2]], name="in_placeholder")
        out_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,1],name='out_placeholder')
        phase = tf.placeholder(tf.bool,name='phase')
        net_class = Models(args)
        net_class.build_model(in_placeholder,phase)

        if run_CPU:
            config = tf.ConfigProto(device_count = {'GPU': 0})
            sess = tf.Session(config=config)
        else:
            sess = tf.Session()

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoints_path)
        if ckpt and ckpt.model_checkpoint_path: 
            #ckpt_path = ckpt.model_checkpoint_path
            ckpt_path = checkpoints_path + 'my_model-' + str(checkpoint)
            saver.restore(sess, ckpt_path)
            
        if save_images:
            output_folder = results_folder + '/' + dataset + '_output/' + str(checkpoint)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
        if compute_analytic and save_images:
            analytic_output_folder = results_folder + '/' + dataset + '_output/analytic'
            if not os.path.exists(analytic_output_folder):
                os.makedirs(analytic_output_folder)
            
        for i in range(len(image_list)):
    
            image_name = image_list[i]
            image = np.array(Image.open(hazy_path + '/' + image_name))
            if dataset == 'HSTS':
                image_number = image_name[:image_name.index(".")]   
            else:
                image_number = image_name[:image_name.index("_")]
            orig_im = np.array(Image.open(orig_path + '/' + image_number + gt_file_type))/255.0
            if (np.shape(orig_im)[2]>3):
                orig_im = orig_im[:,:,:3]
            orig_im = orig_im.astype(np.float64)
    
            if compute_analytic:
                utils.tic()
                J_analytic, loss_analytic[i] = utils.analytic_dehaze(image/255.0, args.strength,\
                        args.dehaze_win, args.dehaze_omega, args.dehaze_thresh, args.dehaze_eps)
                time_analytic[i] = utils.toc()
                J_analytic = J_analytic.astype(np.float64)
                MSE = np.sqrt(np.mean(np.power(orig_im - J_analytic,2)))
                PSNR_analytic[i] = 20*math.log10(1.0/MSE)
                SSIM_analytic[i] = ssim(orig_im,J_analytic,multichannel=True,gaussian_weights=True,sigma=1.5,use_sample_covariance=False) if compute_SSIM else 0
            
                if save_images:
                    Image.fromarray(utils.prep_save(J_analytic)).save(analytic_output_folder + '/' +  image_name[:image_name.index(hazy_file_type)] + gt_file_type)
            
            im_for_tf = np.expand_dims(image/255.0,axis=0)
            utils.tic()
            t_network_fine = np.squeeze(sess.run([net_class.network_out], feed_dict={in_placeholder: im_for_tf, phase:False}))
            J_network_fine = utils.clean_from_hazy(image,t_network_fine,args.dehaze_win,args.dehaze_omega,args.dehaze_thresh)
            time_network[i] = utils.toc()
            
            if compute_loss:
                t_network_fine = np.expand_dims(np.expand_dims(np.squeeze(t_network_fine),axis=0),axis=-1)
                args.image_dim = np.shape(image)
                dehaze_loss = Dehazing_Loss(args)
                loss_op = dehaze_loss._loss(in_placeholder,out_placeholder)
                feed_dict = {in_placeholder: im_for_tf, out_placeholder: t_network_fine, phase:False}
                loss_network[i] = sess.run(loss_op,feed_dict=feed_dict)
            
            if save_images:
                Image.fromarray(utils.prep_save(J_network_fine)).save(output_folder + '/' +  image_name[:image_name.index(hazy_file_type)] + gt_file_type)
                    
            J_network_fine = J_network_fine.astype(np.float64)
            MSE = np.sqrt(np.mean(np.power(orig_im - J_network_fine,2)))
            PSNR_network[i] = 20*math.log10(1.0/MSE)
            SSIM_network[i] = ssim(orig_im,J_network_fine,multichannel=True,gaussian_weights=True, sigma=1.5, use_sample_covariance=False) if compute_SSIM else 0

            if compute_analytic:        
                print ("image (%d/%d) %s, PSNR_analytic: %.4f, PSNR_network: %.4f, SSIM_analytic: %.4f, SSIM_network: %.4f, loss_analytic: %.4f, loss_network: %.4f, time_analytic: %.4f, time_network: %.4f" % \
                       (i+1, len(image_list), image_name,PSNR_analytic[i],PSNR_network[i],SSIM_analytic[i],SSIM_network[i],\
                        loss_analytic[i],loss_network[i],time_analytic[i],time_network[i]))  
            else:
                print ("image (%d/%d) %s, PSNR_network: %.4f, SSIM_network: %.4f, loss_network: %.4f, time_network: %.4f" % \
                       (i+1, len(image_list), image_name,PSNR_network[i],SSIM_network[i],loss_network[i],time_network[i]))
                
        if compute_analytic:
            row = str(checkpoint),'%.5f' % np.mean(PSNR_analytic), '%.5f' % np.mean(PSNR_network), '%.5f' % np.mean(SSIM_analytic),'%.5f' % np.mean(SSIM_network),'%.5f' % \
                np.mean(loss_analytic), '%.5f' % np.mean(loss_network), '%.4f' % np.mean(time_analytic), '%.4f' % np.mean(time_network)
        else:
            row = str(checkpoint),'%.5f' % np.mean(PSNR_network), '%.5f' % np.mean(SSIM_network),'%.5f' % np.mean(loss_network), '%.4f' % np.mean(time_network)
        writer.writerow(row)
        data_file.flush()
        if compute_analytic:
            print ("Mean PSNR: analytic: %.4f, network: %.4f, Mean SSIM: analytic: %.4f, network: %.4f, Mean Loss: analytic: %.4f, network: %.4f, Mean runtime: analytic: %.4f, network: %.4f" % \
                   (np.mean(PSNR_analytic),np.mean(PSNR_network),np.mean(SSIM_analytic),np.mean(SSIM_network),np.mean(loss_analytic),np.mean(loss_network),np.mean(time_analytic),np.mean(time_network)))  
        else: 
            print ("Mean PSNR: network: %.4f, Mean SSIM: network: %.4f, Mean Loss: network: %.4f, Mean runtime: %.4f" % (np.mean(PSNR_network),np.mean(SSIM_network),np.mean(loss_network),np.mean(time_network)))  
        
    data_file.close()
    sess.close()

#%% compute Qualitative results 
if compute_qual:
    
    for checkpoint in qual_checkpoints:
        
        # load saved network parameters and open new session
        tf.reset_default_graph()
        in_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, args.image_dim[2]], name="in_placeholder")
        out_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,1],name='out_placeholder')
        phase = tf.placeholder(tf.bool,name='phase')
        net_class = Models(args)
        net_class.build_model(in_placeholder,phase)
        
        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoints_path)
        if ckpt and ckpt.model_checkpoint_path: 
            ckpt_path = checkpoints_path + 'my_model-' + str(checkpoint)
            saver.restore(sess, ckpt_path)
        
        output_folder = results_folder + '/Qual_output/' + str(checkpoint)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        if compute_analytic:
            analytic_output_folder = results_folder + '/Qual_output/analytic'          
            if not os.path.exists(analytic_output_folder):
                os.makedirs(analytic_output_folder)

        image_list = sorted(os.listdir(qual_path))
    
        for i in range(len(image_list)):
    
            file_name = image_list[i]
            image = Image.open(qual_path + '/' + file_name)
            image_name = file_name[:file_name.index('.')]
            file_type = file_name[file_name.index('.'):]
            image = np.array(image)
            if np.shape(image)[2] > 3:
                image = image[:,:,0:3]
            print ('image (%d/%d)' % (i+1,len(image_list)))
    
            if compute_analytic:
                J_analytic, _ = utils.analytic_dehaze(image/255.0, args.strength,\
                    args.dehaze_win, args.dehaze_omega, args.dehaze_thresh, args.dehaze_eps)
                Image.fromarray(utils.prep_save(J_analytic)).save(analytic_output_folder + "/" + \
                    image_name + file_type)
    
            args.image_dim = np.shape(image)
            dehaze_loss = Dehazing_Loss(args)
            im_for_tf = np.expand_dims(image/255.0,axis=0)
            t_network = sess.run([net_class.network_out], feed_dict={in_placeholder: im_for_tf, phase:False})
            t_network = np.expand_dims(np.expand_dims(np.squeeze(t_network),axis=0),axis=-1)
            J_network = utils.clean_from_hazy(image,np.squeeze(t_network),\
                args.dehaze_win,args.dehaze_omega,args.dehaze_thresh)
            J_network = (J_network - np.min(J_network))/(np.max(J_network)-np.min(J_network))
            
            Image.fromarray(utils.prep_save(J_network)).save(output_folder + "/" + \
                image_name + file_type)
            
    sess.close()