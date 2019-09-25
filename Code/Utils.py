import numpy as np
import scipy as sp
from skimage.util import view_as_windows as viewW
import time
import math

#%% utility to measure runtime
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        #print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
        return tempTimeInterval

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

#%%  utility to display images
def imshow_sub(image,h,title):
    
    if (len(np.shape(image))<=2):
        h.imshow(image,cmap='gray')
    elif np.shape(image)[2] == 1:
        h.imshow(np.squeeze(image),cmap='gray')
    elif np.shape(image)[2] == 3:
        h.imshow(image)
    h.set_title(title)
    h.axes.get_xaxis().set_visible(False)
    h.axes.get_yaxis().set_visible(False)
    
#%% prepare [0,1] float image to save
def prep_save(image):
    
    return (np.squeeze(image)*255).astype(np.uint8)

#%% ***************************** Functions Dealing with Segmentation *********************************
### ***************************************************************************************************    
# compute analytic solution of segmentation energy function
def analytic_seg(x, num_classes, strength, beta):
    
    # inputs:
    # x - image + seeds
    # num_classes - 21 in case of Pascal VOC
    # strength = [1,1] - lambda values of smoothness and data terms in energy function
    # beta - 100/1000 - beta value of weights calculation in energy function
    # output:
    # y - output probabilities of each class
    # loss - loss values of entire energy function
    # smooth_loss, data_loss - individual components of loss function
    
    H = np.size(x,0)
    W = np.size(x,1)
    seeds = x[:,:,:num_classes]
    image = x[:,:,num_classes:]
    y = np.zeros((H*W,num_classes))
    Q = compute_q_matrix(seeds)
    L = compute_laplacian_seg(image,beta)
    x = np.reshape(seeds,(H*W,num_classes))
    b = strength["data_term"] * Q * x
    A = strength["smoothness_term"] * L + strength["data_term"] * Q
    for i in range(1,num_classes):
        y[:,i] = sp.sparse.linalg.cg(A,b[:,i])[0]
    y[:,0] = 1.0 - np.sum(y[:,1:],axis=-1)
    smooth_loss = strength["smoothness_term"]*np.sum(np.diag(np.dot(y.T,L*y)))
    data_loss = strength["data_term"]*np.sum(np.diag(np.dot((y-x).T,Q*(y-x))))
    loss = smooth_loss + data_loss
    y = np.reshape(y,(H,W,num_classes))
    
    return y, loss, smooth_loss, data_loss

#%% compute matrix Q of segmentation energy function
def compute_q_matrix(seeds):
    
    # inputs:
    # seeds - matrix of size [Height, Width, Num_classes], containing mostly zeros with '1' in seed locations of each class
    # ouptut:
    # Q - matrix to be used in energy function
    
    H = seeds.shape[0]
    W = seeds.shape[1]
    num_classes = seeds.shape[-1]
    seeds = np.reshape(seeds,(H*W,num_classes))
    q = np.sum(seeds, axis=-1)
    Q = sp.sparse.coo_matrix((q,(range(H*W),range(H*W))),shape=(H*W,H*W))
    Q = Q.tocsr()
    
    return Q

#%% compute Laplacian of segmentation energy function
def compute_laplacian_seg(image, beta=1000):
    
    # input:
    # image - natural image of size [Height, Width, Num_channels]
    # beta - parameter of weight calculation of the Laplacian matrix
    # output: 
    # L - sparse Laplacian matrix to be used for analytic calculation of energy function
    
    H = np.shape(image)[0]
    W = np.shape(image)[1]
    C = np.shape(image)[2]
    image = np.reshape(image, (H*W,C))

    ind = np.array(range(H*W)).astype(np.int32)
    I_new = np.hstack((ind,ind,ind,ind,ind)).astype(np.int32)
    J_bottom = np.zeros(H*W,dtype=np.int32)
    J_top = np.zeros(H*W,dtype=np.int32)
    J_left = np.zeros(H*W,dtype=np.int32)
    J_right = np.zeros(H*W,dtype=np.int32)
    J_left[ind % W != 0] = ind[ind % W != 0] - 1
    J_left[ind % W == 0] = ind[ind % W == 0] + (W - 1)
    J_right[(ind+1) % W != 0] = ind[(ind+1) % W != 0] + 1
    J_right[(ind+1) % W == 0] = ind[(ind+1) % W == 0] - (W - 1)
    J_top[ind >= W] = ind[ind >= W] - W
    J_top[ind < W] = ind[ind < W] + (H - 1)*W
    J_bottom[ind < W*(H-1)] = ind[ind < W*(H-1)] + W
    J_bottom[ind >= W*(H-1)] = ind[ind >= W*(H-1)] - W*(H-1)
    J_new = np.hstack((ind,J_left,J_right,J_top,J_bottom)).astype(np.int32)
    distances = np.linalg.norm(image[I_new[H*W:],:]-image[J_new[H*W:],:],axis=-1).astype(np.float32)
    sq_distances = np.power(distances,2)
    V_cross = -np.exp(-beta*sq_distances).astype(np.float32)
    V_self = -np.sum(np.reshape(V_cross,(4,H*W)),axis=0).astype(np.float32)
    V_new = np.hstack((V_self,V_cross))
    
    L = sp.sparse.coo_matrix((V_new, (I_new, J_new)), shape=(H*W, H*W))
    L = L.tocsr()
    
    return L

#%% utility function that creates united matrix containing both image and corresponding seeds
def concat_image_seeds(im,orig_seed,seed,num_classes):
    
    # inputs:
    # im - reasized natural input image of size [Height,Width,Num_channels]
    # orig_seed - seeds in original dimension (before resize) 
    # seed - current seed after resize of size [Height,Width,Num_classes]
    # num_classes - classes in dataset
    # output:
    # image_and_seeds - united strcuture for both image and seeds of size: [Height,Width,Num_Classes+Num_Channels]
    
    dims = np.shape(im)
    image_and_seeds = np.zeros((dims[0],dims[1],dims[2]+num_classes))
    image_and_seeds[:,:,num_classes:] = im
    classes_in_image = np.unique(orig_seed)[1:]
    for c in classes_in_image:
        seed_coords = np.nonzero(seed == c)
        image_and_seeds[seed_coords[0],seed_coords[1],c-1] = 1
        
    return image_and_seeds

#%% utility function to compute confusion table for mIOU calculation of Pascal VOC 2012
def conf_counts_add(seg,gt,num_classes):
    
    # input: 
    # seg - predicted segmentation of size [Height,Width]
    # gt - ground truth segmentation of size [Height,Width]
    # num_classes - number of classes in dataset
    # output:
    # hs - incremental addition of current image to confusion table
    
    gt = np.array(gt)
    locs = np.nonzero(gt < 255)
    seg = np.array(seg).astype(np.uint16)
    sumim = gt + seg*num_classes 
    [hs,_] = histc(sumim[locs],np.array(range(num_classes**2)))
    
    return hs

#%% utility function to compute confusion table for mIOU calculation of Pascal VOC 2012
def histc(X, bins):
    
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return [r, map_to_bins]

#%% save segmentation to disk
def save_seg(save_flag,seg,palette,path):
    
    if save_flag:
        seg.putpalette(palette)
        seg.save(path)
    return

#%% display segmentations
def imshow_sub_seg(image,h,title):
    
    if (len(np.shape(image))<=2):
        h.imshow(image)
    elif np.shape(image)[2] == 1:
        h.imshow(np.squeeze(image))
    elif np.shape(image)[2] == 3:
        h.imshow(image)
    h.set_title(title)
    h.axes.get_xaxis().set_visible(False)
    h.axes.get_yaxis().set_visible(False)

#%% final calculation of confusion matrix for entire Pascal VOC 2012
def conf_mat(conf_counts_init,num_classes):
    
    # input:
    # conf_counts_init - confusion matrix after addition of contributions of each image/segmentation
    # num_classes - number of classes in dataset
    # output: 
    # accuracies - mIOU scores for each class
    
    conf_mat = np.reshape(conf_counts_init,(num_classes,num_classes),order='F')
    accuracies = np.zeros(num_classes)
    for i in range(num_classes):
        gtj = np.sum(conf_mat[i,:])
        resj = np.sum(conf_mat[:,i])
        gtjresj = conf_mat[i,i]
        accuracies[i] = gtjresj/(gtj+resj-gtjresj)
    
    return accuracies

#%% ***************************** Functions Dealing with Image Matting *********************************
### *************************************************************************************************** 
# analytic calculation of matting energy function
def analytic_matte(x, num_classes, strength, eps, f_size):
    
    # inputs:
    # x - input image + seeds of size [Height,Width,Num_classes+Num_channels]
    # num_classes - number of classes = 2, foreground and background
    # strength - relative strength of smoothness and data terms in matting energy function
    # eps - epsilon parameter in weights calculation of the Matting Laplcian
    # f_size - size of window - default is [3,3]
    # outputs:
    # y - output alpha matte of size [Height,Width,1]
    # loss, smooth_loss, data_loss - overall loss function value and individual smoothness and data components
    
    H = np.size(x,0)
    W = np.size(x,1)
    seeds = x[:,:,:num_classes]
    image = np.squeeze(x[:,:,num_classes:])
    y = np.zeros((H*W,num_classes))
    Q = compute_q_matrix(seeds)
    L = compute_laplacian_matte(image, eps, [f_size,f_size])
    x = np.reshape(seeds,(H*W,num_classes))
    x = x[:,1] # we are only interested in the foreground in this case
    b = strength["data_term"] * Q * x
    A = strength["smoothness_term"] * L + strength["data_term"] * Q
    y = sp.sparse.linalg.cg(A,b)[0]
    smooth_loss = strength["smoothness_term"]*np.dot(y.T,L*y)
    data_loss = strength["data_term"]*np.dot((y-x).T,Q*(y-x))
    loss = smooth_loss + data_loss
    y = np.reshape(y,(H,W))
    y = np.minimum(np.maximum(y, 0), 1)
    
    return y, loss, smooth_loss, data_loss

#%% compute Matting Laplacian matrix for analytic solution
def compute_laplacian_matte(image, eps=1e-5, f_size=[3,3]):
    
    # inputs: 
    # image - natural input image of size [Height,Width,Num_channels]
    # eps - epsilon parameter in weights calculation
    # f_size - window size in weight calcaultion
    # output: 
    # L - sparse Matting Laplcian matrix
    
    H = np.shape(image)[0]
    W = np.shape(image)[1]
    C = np.shape(image)[2] # num channels = 3
    patch_size = f_size[0]*f_size[1]
    num_patches = (H-(f_size[0]-1))*(W-(f_size[1]-1))
    inds = np.arange(H*W).reshape(H,W)
    inds_win = viewW(inds,f_size).reshape(num_patches,patch_size)
    I = np.repeat(inds_win,patch_size,axis=1)
    J = np.tile(inds_win,(1,patch_size))
    image_win = viewW(image,(f_size[0],f_size[1],C)).reshape(num_patches,patch_size,C)
    image_mean = np.mean(image_win,axis=1,keepdims=True)
    image_var = (np.matmul(np.transpose(image_win,(0,2,1)),image_win)/patch_size - \
                 np.matmul(np.transpose(image_mean,(0,2,1)),image_mean))
    matrix_to_invert = (eps/patch_size)*np.eye(C) + image_var
    var_fac = np.linalg.inv(matrix_to_invert)
    X = np.matmul(image_win - image_mean, var_fac)
    V = np.matmul(X, np.transpose(image_win - image_mean, (0,2,1)))
    weights = (1.0/patch_size)*(1 + V)
    V = np.eye(patch_size) - weights
    V = V.reshape((-1,patch_size*patch_size))
    
    L = sp.sparse.coo_matrix((V.ravel(), (I.ravel(), J.ravel())), shape=(H*W, H*W))
    L = L.tocsr()
    
    return L

#%% utility function to compute MSE (mean square error) metric for matting evaluation
def compute_MSE_with_trimap(pred,gt,trimap):
    
    # inupts:
    # pred - predicted alpha matte of size [Height,Width,1]
    # gt - ground truth alpha matte
    # trimap - input trimap 
    # output:
    # mse value
    
    if (np.max(gt) > 1):
        gt = gt/255.0
    if len(np.shape(trimap)) > 2:
        trimap = trimap[:,:,0] + trimap[:,:,1]
        unknown_pix = np.equal(trimap,0)
    else:
        unknown_pix = np.equal(trimap,128)
    sum_unknown_pix = np.sum(unknown_pix.astype(np.uint8))
    se = (pred - gt)**2
    sum_se = np.sum(se[unknown_pix])
    mse = sum_se/float(sum_unknown_pix)
    
    return mse

#%% utility function to compute SAD (sum of absolute differences) metric for matting evaluation
def compute_SAD_with_trimap(pred,gt,trimap):
    
    # inupts:
    # pred - predicted alpha matte of size [Height,Width,1]
    # gt - ground truth alpha matte
    # trimap - input trimap 
    # output:
    # sad value
    
    if (np.max(gt) > 1):
        gt = gt/255.0
    if len(np.shape(trimap)) > 2:
        trimap = trimap[:,:,0] + trimap[:,:,1]
        unknown_pix = np.equal(trimap,0)
    else:
        unknown_pix = np.equal(trimap,128)
    #sum_unknown_pix = np.sum(unknown_pix.astype(np.uint8))
    se = np.abs(pred-gt)
    sum_se = np.sum(se[unknown_pix])
    sad = sum_se
    
    return sad
    
#%% ***************************** Functions Dealing with Image Matting *********************************
### *************************************************************************************************** 
# compute analytic solution of dehazing energy function
def analytic_dehaze(image, strength, win_size=[15,15], omega=0.95, t0=0.1, epsilon=1e-6):
    
    # inputs:
    # image - input hazy image of size [Height,Width,Num_channels]
    # strength - relative strength of smoothness and data components of energy function
    # win_size - DCP window size, default is [15,15]
    # omega - DCP omega parameter (residual haze amount), default is 0.95
    # t0 - DCP threshold parameter, default is 0.1
    # epsilon - DCP epsilon parameter (from matting energy function)
    # outputs:
    # J - reconstructed dehazed image of size [Height,Width,Num_channels]
    # loss - overall loss value of dehazing energy function
    
    H = np.size(image,0)
    W = np.size(image,1)
    t_coarse, atmo_vec = compute_transmittance(image,win_size,omega)
    t_coarse = np.squeeze(np.reshape(t_coarse,(H*W,-1)))
    L = compute_laplacian_matte(image,epsilon,[3,3])
    Q = sp.sparse.eye(H*W)
    b = strength["data_term"] * Q * t_coarse
    A = strength["smoothness_term"] * L + strength["data_term"] * Q
    t_fine = sp.sparse.linalg.cg(A,b)[0]
    smooth_loss = strength["smoothness_term"]*np.dot(t_fine.T,L*t_fine)
    data_loss = strength["data_term"]*np.dot((t_fine-t_coarse).T,(t_fine-t_coarse))
    loss = smooth_loss + data_loss
    t_fine = np.reshape(t_fine,(H,W))    
    t_fine[t_fine<t0] = t0
    J = (image - atmo_vec)/np.stack((t_fine,t_fine,t_fine),axis=-1) + atmo_vec
    J = (J - np.min(J))/(np.max(J)-np.min(J))

    return J,loss

#%% compute initial DCP-based coarse transmission map
def compute_transmittance(image, win_size, omega):
    
    # image - input hazy image of size [Height,Width,Num_channels]
    # win_size - DCP window size, default is [15,15]
    # omega - DCP omega (residual haze) parameter, default is 0.95
    # outputs:
    # t - output coarse transmission map of size [Height,Width]
    # A - estimation of airlight vector of size [3,1]
    
    H = np.shape(image)[0]
    W = np.shape(image)[1]
    C = np.shape(image)[2] # number of channels = 3
    patch_size = win_size[0]*win_size[1]
    padded_image = np.pad(image, (((win_size[0]-1)/2,(win_size[0]-1)/2),((win_size[1]-1)/2,(win_size[1]-1)/2),(0,0)), 'edge')
    num_patches = H*W
    image_win = viewW(padded_image,(win_size[0],win_size[1],C)).reshape(num_patches,patch_size,C)
    initial_DCP = np.min(np.reshape(image_win,(num_patches,-1)),axis=1)
    DCP_sorted_inds = np.argsort(initial_DCP)[::-1]
    brightest_pixel_coords = DCP_sorted_inds[:int(0.001*num_patches)] 
    reshaped_image = np.reshape(image,(H*W,C))
    brightest_pixels = reshaped_image[brightest_pixel_coords,:]
    A = np.max(brightest_pixels,axis=0)
    DCP = np.min(np.reshape(image_win/A,(num_patches,-1)),axis=1)
    t = np.reshape(1 - omega*DCP,(H,W))
    
    return t, A

#%% given a hazy image and predicted, refined transmission map, compute the dehazed reconstruction 
def clean_from_hazy(image, t_fine, win_size, omega, t0):
    
    # image - input hazy image of size [Height,Width,Num_channles]
    # t_fine - refined, predicted transmission map, of size [Height,Width]
    # win_size - DCP window size, default is [15,15]
    # omega - DCP omega (residual haze) parameter, default is 0.95
    # t0 - DCP threshold parameter, defauls is 0.1
    # output:
    # J - reconstructed dehazed image of size [Height,Width,Num_channels]
    
    H = np.shape(image)[0]
    W = np.shape(image)[1]
    C = np.shape(image)[2] # number of channels = 3
    patch_size = win_size[0]*win_size[1]
    padded_image = np.pad(image, (((win_size[0]-1)/2,(win_size[0]-1)/2),((win_size[1]-1)/2,(win_size[1]-1)/2),(0,0)), 'edge')
    num_patches = H*W
    image_win = viewW(padded_image,(win_size[0],win_size[1],C)).reshape(num_patches,patch_size,C)
    initial_DCP = np.min(np.reshape(image_win,(num_patches,-1)),axis=1)
    DCP_sorted_inds = np.argsort(initial_DCP)[::-1]
    brightest_pixel_coords = DCP_sorted_inds[:int(0.001*num_patches)]
    reshaped_image = np.reshape(image,(H*W,C))
    brightest_pixels = reshaped_image[brightest_pixel_coords,:]
    A = np.max(brightest_pixels,axis=0)
    t_fine[t_fine<t0] = t0
    ##### added for speedup at 12.9.19
    image = image/255.0
    A = A/255.00
    J = (image - A)/np.stack((t_fine,t_fine,t_fine),axis=-1) + A
    J = (J - np.min(J))/(np.max(J)-np.min(J))
    
    return J

#%% 
def compute_PSNR(pred,gt):
    if np.max(gt) > 1:
        gt = gt/255.0
    diff = pred - gt
    MSE = np.sqrt(np.mean(np.power(diff,2)))
    PSNR = 20*math.log10(1.0/MSE)
    return PSNR