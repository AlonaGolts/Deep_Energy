import numpy as np
import shutil
#import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import Utils as utils
import train_utils as train_utils
import os.path
import datetime
import csv
import os
from PIL import Image
from os.path import dirname, abspath
from Segmentation_Loss import Segmentation_Loss
from Matting_Loss import Matting_Loss
from Dehazing_Loss import Dehazing_Loss
from Models import Models

#%%
def get_batch(i, inputs, mode):
      
    size_data = np.shape(inputs)[0]
    start_ind = (i * args.batch_size) % size_data
    end_ind = ((i + 1) * args.batch_size) % size_data
    num_dims = len(np.shape(inputs))
    if start_ind > end_ind and mode == 'eval': # if eval mode and end of dataset, don't cycle to beginning of dataset
        end_ind = size_data 
    if end_ind > start_ind:
        if num_dims == 4: # input batch for training
            input_batch = inputs[start_ind:end_ind,:,:,:]
        elif num_dims == 3: # ground truth batch for evaluation
            input_batch = inputs[start_ind:end_ind,:,:]
    else:  # this can only be the case for mode='train' not mode='eval'
        input_batch = np.concatenate([inputs[start_ind:,:,:,:], inputs[:end_ind,:,:,:]],0)
    input_batch = input_batch.astype(np.float32)
    if (args.app == 'dehaze'):
        return input_batch
    else:
        num_channels = np.shape(inputs)[-1] - args.num_classes
        if (num_channels == 3): # if color image
            #input_batch[:,:,:,num_classes:] = input_batch[:,:,:,num_classes:]
            input_batch[:,:,:,args.num_classes:] = input_batch[:,:,:,args.num_classes:]/255.0 - args.rgb_avg
    return input_batch

#%%
def evaluate_model(images, gts):
    
    steps_per_epoch = int(np.ceil(np.float32(np.shape(images)[0]) / args.batch_size))
    (sum_loss, sum_criterion) = (0,0)
    conf_counts = np.zeros(args.num_classes**2)
      
    for i in xrange(steps_per_epoch):
        next_batch = get_batch(i,images,'eval')
        feed_dict = {input_placeholder: next_batch, phase: False}
        y_batch, curr_loss = sess.run([output_from_network,loss], feed_dict = feed_dict)
        sum_loss+= curr_loss
        if len(gts)>0:
            gt_batch = get_batch(i,gts,'eval')
            for i in range(np.shape(next_batch)[0]):
                curr_output = np.squeeze(y_batch[i,:,:,:])
            
                if args.app == 'seg':
                    #y_seg = utils.im_segment(curr_output,args.seed,args.seg_type,args.seg_thresh).astype(np.uint16)
                    y_seg = np.argmax(curr_output, axis=-1).astype(np.uint16)
                    gt_seg =  gt_batch[i,:,:].astype(np.uint16)
                    locs = np.nonzero(gt_seg < 255)
                    sumim = y_seg + gt_seg*args.num_classes
                    [hs,_] = utils.histc(sumim[locs],np.array(range(args.num_classes**2)))
                    conf_counts = conf_counts + hs
                
                if args.app == 'matte':
                    curr_image = np.squeeze(next_batch[i,:,:,:args.num_classes])
                    curr_gt = gt_batch[i,:,:]
                    sum_criterion+= utils.compute_MSE_with_trimap(curr_output,curr_gt,curr_image)
                
                if args.app == 'dehaze':
                    curr_image = np.squeeze(next_batch[i,:,:,:])
                    curr_gt = gt_batch[i,:,:,:]
                    curr_J = utils.clean_from_hazy(curr_image, curr_output, args.dehaze_win, args.dehaze_omega, args.dehaze_thresh)
                    sum_criterion+= utils.compute_PSNR(curr_J,curr_gt)
            
            if (args.app == 'seg'):
                conf_mat = np.reshape(conf_counts,(args.num_classes,args.num_classes),order='F')
                accuracies = np.zeros(args.num_classes)
                for i in range(args.num_classes):
                    gtj = np.sum(conf_mat[i,:])
                    resj = np.sum(conf_mat[:,i])
                    gtjresj = conf_mat[i,i]
                    accuracies[i] = gtjresj/(gtj+resj-gtjresj)
                    sum_criterion = np.mean(accuracies)
            elif (args.app == 'dehaze') or (args.app == 'matte'):
                sum_criterion = sum_criterion/np.shape(images)[0]
                
    return sum_loss/np.shape(images)[0], sum_criterion

#%%
config_file_path = 'params.ini'
args = train_utils.parse_params(config_file_path)  

parent_dir = str(dirname(dirname(abspath(__file__))))

# create output directory
curr_time = str(datetime.datetime.now())
output_path = parent_dir + args.output_dir + '/' + args.app + '_' + curr_time[:curr_time.find('.')]
if not os.path.exists(output_path):
    os.makedirs(output_path)

# save configuration file in new results folder
shutil.copyfile('params.ini', output_path + '/params.ini')  

# read data and call loss function constructors
input_path = parent_dir + '/HDF5_files/' + args.hdf5_files
f_HDF5 = h5py.File(input_path,"r+")
x_train =  f_HDF5["train"]
args.image_dim = (np.shape(x_train)[1],np.shape(x_train)[2],np.shape(x_train)[3]-args.num_classes)

if args.app == 'seg':
    (x_test, seg_test) = (f_HDF5["test"],f_HDF5["test_gt"])
    app_loss = Segmentation_Loss(args)
elif args.app == 'dehaze':
    (x_test, gt_test) = (f_HDF5["test"],f_HDF5["test_gt"])
    app_loss = Dehazing_Loss(args)
elif args.app == 'matte':
    (x_test, x_test_1, x_test_2, gt_test) = (f_HDF5["val"],f_HDF5["test_t2"],f_HDF5["test_t1"],f_HDF5["test_gt"])
    app_loss = Matting_Loss(args)

print("Constructing network...")
net_class = Models(args)
tf.reset_default_graph()
input_placeholder = tf.placeholder(tf.float32, (None,) + x_train.shape[1:], 'images_with_seeds')
phase = tf.placeholder(tf.bool,name='phase')
output_from_network = net_class.build_model(input_placeholder,phase)
loss = app_loss._loss(input_placeholder, output_from_network)        

samples_in_epoch = int(np.ceil(np.shape(x_train)[0]/np.float32(args.batch_size)))
num_iter = args.num_epochs*samples_in_epoch + 1

global_step = tf.Variable(0, trainable=False)

if args.train_schedule == 'exp': # decrease learning exponentially
    learning_rate = tf.train.exponential_decay(args.init_lr, global_step,samples_in_epoch*args.decay_every,\
        args.decay_factor, staircase=True)
elif args.train_schedule == 'plateau': # decrease learning rate when plateau is reached
    lr_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
    learning_rate = lr_placeholder
    
print("Setting optimizer...")
optimizer = tf.train.AdamOptimizer(learning_rate, name ='Adam')
        
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this is for batch normalization update during training
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step=global_step)

print("Setting log file...")
loss_path = output_path + "/loss.csv"
res_file  = open(loss_path, "wb")
writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
header_dict = {'seg': ('epoch', 'test_IOU', 'test_loss', 'lr'), \
               'matte': ('epoch', 'test_loss' , 'test_MSE_t1', 'test_MSE_t2', 'lr'), \
               'dehaze': ('epoch', 'test_loss', 'test_PSNR', 'lr')}
header_row = header_dict[args.app]
writer.writerow(header_row)
    
print("Opening a new session...")
sess = tf.Session()

print("Setting up Saver...")
saver = tf.train.Saver(max_to_keep=args.num_epochs)

print("Initializaing variables...")
sess.run(tf.global_variables_initializer())
    
epoch = 0
    
# load model if the checkpoint exists
ckpt = tf.train.get_checkpoint_state(output_path+'/checkpoints/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...")
    model_name = str(ckpt.model_checkpoint_path)
    model_name = model_name[str.find(model_name,args.checkpoint_name):]
    epoch = int(model_name[str.find(model_name,'-')+1:])    

curr_lr = args.init_lr
if args.train_schedule == 'plateau':    
    last_loss = 1000000
    last_diff = -1
    num_changes = 0
    curr_cooltime = args.cool_time
    
print("TRAINING BEGINS")
for i in xrange(num_iter):
    input_batch = get_batch(i,x_train,'train')
    if args.train_schedule == 'plateau':
        feed_dict = {input_placeholder: input_batch, phase: True, lr_placeholder: curr_lr}
    else:
        feed_dict = {input_placeholder: input_batch, phase: True}
        
    sess.run(train_op, feed_dict = feed_dict)
    
    # save model checkpoint
    if (i % (samples_in_epoch*args.epochs_to_checkpoint) == 0) & (i>0):
        saver.save(sess,output_path + '/checkpoints/' + args.checkpoint_name,global_step=epoch+1)
            
    # print loss data every self.print_iter iterations. Don't print if self.print_iter = -1
    if (args.print_iter > 0) and (i % args.print_iter == 0):
        _, loss_value, loss_smooth, loss_data, net_out = sess.run([train_op, loss, \
                app_loss.loss_smooth, app_loss.loss_data,output_from_network], feed_dict = feed_dict)
        print ("iteration: %d, total_loss: %.2f, smoothness_loss: %.2f, data_loss: %.2f" % \
               (i, loss_value/args.batch_size, loss_smooth/args.batch_size, loss_data/args.batch_size))
        
    if (args.display_every > 0) and (i % args.display_every == 0):
        palette = Image.open(parent_dir + '/Datasets/Seg/VOC2012_val/palette/2007_000039.png').getpalette()
        train_utils.show_train_images(args, input_batch[0,:,:,:],net_out[0,:,:,:], palette)
            
    # Evaluate train and test loss and mIOU for each epoch
    if (i % (samples_in_epoch) == 0) & (i>0):
#    if (i>0):       
        epoch+= 1
            
        if args.app == 'seg':            
            test_loss,test_IOU = evaluate_model(x_test,seg_test)
            print("epoch: %d, test_mIOU: %.3f, avg_test_loss: %.3f, lr: %.5f" % (epoch,test_IOU,test_loss,curr_lr))
            row = str(epoch),'%.5f' % test_IOU, '%.5f' % test_loss, '%.5f' % curr_lr
                
        elif args.app == 'matte':
            test_loss,_ = evaluate_model(x_test,[])
            test_loss_1, test_MSE_t1 = evaluate_model(x_test_1,gt_test)
            test_loss_2, test_MSE_t2 = evaluate_model(x_test_2,gt_test)
            print("epoch: %d, avg_test_loss: %.3f, test_MSE_t1: %.3f, test_MSE_t2: %.3f, lr: %.5f" % (epoch,test_loss,test_MSE_t1,test_MSE_t2,curr_lr))
            row = str(epoch), '%.5f' % test_loss, '%.5f' % test_MSE_t1, '%.5f' % test_MSE_t2, '%.5f' % curr_lr
            
        elif args.app == 'dehaze':
            test_loss, test_PSNR = evaluate_model(x_test,gt_test)
            print("epoch: %d, avg_test_loss: %.3f, avg_test_PSNR: %.4f, lr: %.5f" % (epoch,test_loss,test_PSNR,curr_lr))
            row = str(epoch), '%.5f' % test_loss, '%.5f' % test_PSNR, '%.5f' % curr_lr
            
        # update learning rate when a plateau in test is reached
        if args.train_schedule == 'plateau': 
            curr_diff = test_loss - last_loss
            if (np.sign(curr_diff) != np.sign(last_diff)) and (curr_cooltime <= 0):
                num_changes+= 1
            if num_changes >= args.max_sign_change:
                curr_lr = curr_lr/np.sqrt(args.lr_factor)
                num_changes = 0
                curr_cooltime = args.cool_time
            else:
                curr_cooltime-= 1
            last_loss = test_loss
            last_diff = curr_diff
                
        writer.writerow(row)
        res_file.flush()
            
res_file.close()