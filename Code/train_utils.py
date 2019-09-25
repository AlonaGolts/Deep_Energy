import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import configargparse
import Utils as utils

def parse_params(config_file_path = 'params.ini'):
    
    parser = configargparse.ArgumentParser(default_config_files=[config_file_path])

    parser.add_argument('--seed', type = int, default=3, help='random seed selection, best is seed=3')
    parser.add_argument('--app', type=str, default = 'dehaze', help='seg/matte/dehaze')
    parser.add_argument('--dehaze_hdf5', type=str, default = 'RTTS_aug_4_OTS.hdf5', help='hdf5 file for dehazing')
    parser.add_argument('--seg_hdf5', type=str, default = 'Pascal_VOC_He_Seeds_128_aug_4_old.hdf5', help='hdf5 file for segmentation')
    parser.add_argument('--matte_hdf5', type=str, default = 'Pascal_VOC_Matte_128_aug_1.hdf5', help='hdf5 file for matting')
    parser.add_argument('--lr_dehaze', type = float, default=3e-4, help='dehazing initial learning rate')
    parser.add_argument('--lr_matte', type = float, default=0.01, help='matting initial learning rate')
    parser.add_argument('--lr_seg', type = float, default=0.01, help='segmentation initial learning rate')
    parser.add_argument('--decay_every', type = int, default=3, help='decay every X epochs')
    parser.add_argument('--decay_factor', type = float, default=0.96, help='exponential decay factor [0.9,0.99]')
    parser.add_argument('--num_epochs', type = int, default=3, help='number of epochs over the training data')
    parser.add_argument('--batch_size', type = int, default=24, help='batch size, previous = 24')
    parser.add_argument('--max_sign_change', type = int, default=2, help='learning schedule parameter')
    parser.add_argument('--cool_time', type = int, default=2, help='learning schedule parameter')
    parser.add_argument('--lr_factor', type = float, default=2, help='learning schdule parameter')
    parser.add_argument('--model_type', type=str, default = 'conv_skip', help='model name: conv_skip')
    parser.add_argument('--num_reg_conv', type = int, default=2, help='number of regular convolutions in each dilated residual block, minimum = 1')
    parser.add_argument('--num_net_blocks', type = int, default=6, help='number of dilated residual blocks -> max rate = 2^(num_net_blocks-1)')
    parser.add_argument('--filter_size', type = int, default=3, help='filter size: defaults is 3x3')
    parser.add_argument('--num_filters', type = int, default=32, help='width of each conv layer')
    parser.add_argument('--padding_type', type = str, default='REFLECT', help='REFLECT/None')
    parser.add_argument('--epochs_to_checkpoint', type = int, default=1, help='number of epochs until saved checkpoint')
    parser.add_argument('--print_iter', type = int, default=10, help='print every X iterations')
    parser.add_argument('--checkpoint_name', type = str, default='my_model', help='initial name of checkpoint')
    parser.add_argument('--output_dir', type = str, default='\Results', help='results output path')
    parser.add_argument('--power', type = float, default=2.0, help='power of both terms in energy function')
    parser.add_argument('--seg_lambda_1', type = float, default=10.0, help='smoothness term multiplier in seeded seg')
    parser.add_argument('--seg_lambda_2', type = float, default=100.0, help='data term multiplier in seeded seg')
    parser.add_argument('--beta', type = float, default=1000, help='beta for seeded segmentation energy function')
    parser.add_argument('--num_neighbors', type = int, default=4, help='4-neighborhood for seeded segmentation')         
    parser.add_argument('--matte_lambda_1', type = float, default=1.0, help='matting term mulitplier in matting')         
    parser.add_argument('--matte_lambda_2', type = float, default=1.0, help='data term mulitplier in matting')         
    parser.add_argument('--matte_win_size', type = int, default=3, help='window size for matting application')
    parser.add_argument('--matte_eps', type = float, default=1e-5, help='epsilon value for image matting application')  
    parser.add_argument('--dehaze_lambda_1', type = float, default=1.0, help='matting term mulitplier in matting')  
    parser.add_argument('--dehaze_lambda_2', type = float, default=1e-4, help='data term mulitplier in matting')  
    parser.add_argument('--dehaze_omega', type = float, default=0.95, help='omega parameter in dehazing equation')  
    parser.add_argument('--dehaze_thresh', type = float, default=0.1, help='threshold parameter in dehazing equation')  
    parser.add_argument('--dehaze_win_size', type = int, default=15, help='window size in DCP algorithm')           
    parser.add_argument('--dehaze_eps', type = float, default=1e-6, help='epsilon value for image dehazing application')  
    parser.add_argument('--display_every', type = int, default=50, help='display training images every X iterations')  
    
    args = parser.parse_args()
    
    hdf5_files_dict = {'seg': args.seg_hdf5, 'matte': args.matte_hdf5, 'dehaze':args.dehaze_hdf5}
    rgb_avg_dict = {'seg': [0.46,0.44,0.40], 'matte': [0.46,0.44,0.40], 'dehaze': [0,0,0]}
    num_classes_dict  = {'seg': 21, 'matte': 2, 'dehaze': 0}
    num_outputs_dict = {'seg': 21, 'matte': 1, 'dehaze': 1}
    use_softmax_dict = {'seg': True, 'matte': False, 'dehaze': False}
    lambda_1 = {'seg': args.seg_lambda_1, 'matte': args.matte_lambda_1, 'dehaze': args.dehaze_lambda_1}
    lambda_2 = {'seg': args.seg_lambda_2, 'matte': args.matte_lambda_2, 'dehaze': args.dehaze_lambda_2}
    train_schedule_dict = {'seg': 'plateau', 'matte': 'plateau', 'dehaze': 'exp'}
    init_lr_dict = {'seg': args.lr_seg, 'matte': args.lr_matte, 'dehaze': args.lr_dehaze}
    args.rgb_avg = rgb_avg_dict[args.app]
    args.power = {"smoothness_term": args.power,"data_term": args.power}
    args.strength = {"smoothness_term": lambda_1[args.app], "data_term": lambda_2[args.app]}
    args.num_classes = num_classes_dict[args.app]
    args.num_outputs = num_outputs_dict[args.app]
    args.use_softmax = use_softmax_dict[args.app]
    args.dehaze_win = [args.dehaze_win_size, args.dehaze_win_size]
    args.matte_win = [args.matte_win_size, args.matte_win_size]
    args.hdf5_files = hdf5_files_dict[args.app]
    args.train_schedule = train_schedule_dict[args.app]
    args.init_lr = init_lr_dict[args.app]

    return args

def show_train_images(args, input_im, output_im, palette=None):
    
    if args.app == 'seg':
        fig = plt.figure(figsize=(16,8))
        plt.subplot(131)
        plt.imshow(input_im[:,:,args.num_classes:] + args.rgb_avg)
        plt.title('input image')
        plt.subplot(132)
        plt.imshow(np.squeeze(output_im[:,:,0]),'gray')
        plt.title('output probability')
        #net_seg = utils.im_segment(output_im,args.seed,args.seg_type,args.seg_thresh)
        net_seg = np.argmax(output_im,axis=-1)
        net_seg = Image.fromarray(net_seg.astype(np.uint8))
        net_seg.putpalette(palette)
        plt.subplot(133)
        plt.imshow(net_seg,'gray')   
        plt.title('output seg')
        plt.show()
    if args.app == 'matte':
        fig = plt.figure(figsize=(10,8))
        plt.subplot(121)
        plt.imshow(input_im[:,:,args.num_classes:] + args.rgb_avg)
        plt.title('input image')
        plt.subplot(122)
        plt.imshow(np.squeeze(output_im),'gray')
        plt.title('output matte')
        plt.show()
    if args.app == 'dehaze':
        fig = plt.figure(figsize=(10,8))
        plt.subplot(121)
        plt.imshow(input_im[:,:,args.num_classes:])
        plt.title('input image')
        dehazed_im = utils.clean_from_hazy(np.squeeze(input_im[:,:,args.num_classes:]), np.squeeze(output_im), args.dehaze_win, args.dehaze_omega, args.dehaze_thresh)
        plt.subplot(122)
        plt.imshow(dehazed_im)
        plt.title('output_image')
        plt.show()
        
