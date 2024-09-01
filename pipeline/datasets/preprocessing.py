from typing import Any
import numpy as np
from PIL import Image
from numpy.core.defchararray import add
from numpy.lib.arraysetops import isin
from torchvision import transforms, datasets
from torchvision.transforms import v2
import torch
from scipy import signal
from ..utils import disp_sar

def symetrisation_patch(im):
    S = np.fft.fftshift(np.fft.fft2(im))
    p = np.zeros((S.shape[0])) 
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i,:]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p)*np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(),p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1-1)/2))%p.shape[0]+int(p.shape[0]/2)
    p2_1 = np.roll(p,shift_az_1)
    shift_az_2 = int(round(-(d1-1-p.shape[0])/2))%p.shape[0]+int(p.shape[0]/2)
    p2_2 = np.roll(p,shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2*p.shape[0])
    test_1 = np.sum(window*p2_1)
    test_2 = np.sum(window*p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1>=test_2:
        p2 = p2_1
        shift_az = shift_az_1/p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2/p.shape[0]
    S2 = np.roll(S,int(shift_az*p.shape[0]),axis=0)

    q = np.zeros((S.shape[1])) # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:,j]))
    sq = q[::-1]
    #correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q)*np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(),q.shape[0])
    d2=d2[0]
    shift_range_1 = int(round(-(d2-1)/2))%q.shape[0]+int(q.shape[0]/2)
    q2_1 = np.roll(q,shift_range_1)
    shift_range_2 = int(round(-(d2-1-q.shape[0])/2))%q.shape[0]+int(q.shape[0]/2)
    q2_2 = np.roll(q,shift_range_2)
    window_r = signal.gaussian(q.shape[0], std=0.2*q.shape[0])
    test_1 = np.sum(window_r*q2_1)
    test_2 = np.sum(window_r*q2_2)
    if test_1>=test_2:
        q2 = q2_1
        shift_range = shift_range_1/q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2/q.shape[0]

    Sf = np.roll(S2,int(shift_range*q.shape[0]),axis=1)
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    return ima2

def filter_norm(norm,n_ch):
    if n_ch == 2 :
        norm_ = np.array([[norm[0][0],norm[0][3]],[norm[1][0],norm[1][3]]]) 
    elif n_ch == 3 :
        norm_ = np.array([[norm[0][0],norm[0][1],norm[0][3]],[norm[1][0],norm[1][1],norm[1][3]]]) 
    else :
        norm_ = norm
    return norm_

class ToTensor():
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, im):  
        assert isinstance(im,np.ndarray)      
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = im.transpose((2, 0, 1))
        return torch.from_numpy(im)

class numpy_resize():
    def __init__(self,size):
        self.r = transforms.Resize((size, size),interpolation=transforms.InterpolationMode.NEAREST_EXACT)
    def __call__(self,im:np.ndarray):
        squeeze = False
        sh = im.shape
        if len(sh) == 2:
            im = im[:,:,np.newaxis]
            squeeze = True
        assert len(sh) == 3
        im_t = im.transpose((2, 0, 1))
        im_t = torch.from_numpy(im_t)
        im_t = self.r(im_t)
        im_t = im_t.permute(1, 2, 0).numpy()
        if squeeze:
            im_t = np.squeeze(im_t)
        return im_t
    
class normalization():
    def __init__(self,m,M) -> None:
        self.m_ = m
        self.M_ = M
        
    def __call__(self,im):
        assert isinstance(im,np.ndarray)
        log_im = np.log(np.abs(im)+np.spacing(1))
        num = log_im - self.m_
        den = self.M_-self.m_
        norm = num/den
        norm = np.clip(norm,0,1)
        return (norm).astype(np.float32)

class normalization_():
    def __init__(self,device='cuda') -> None:
        self.device = device
    def __call__(self,im,m,M):

            if isinstance(im,np.ndarray):
                log_im = np.log(np.abs(im)+np.spacing(1))
                log_im = (log_im - m)/(M-m)
                log_im = np.clip(log_im,0,1)
                return (log_im).astype(np.float32)
            elif torch.is_tensor(im):
                if isinstance(m,np.ndarray) and isinstance(M,np.ndarray):
                    min_ = torch.from_numpy(m).to(self.device)
                    max_ = torch.from_numpy(M).to(self.device)
                elif isinstance(m,(float,int)) and isinstance(M,(float,int)):
                    min_ = m
                    max_ = M
                else :
                    print("normalization parameter should be a float or an array")
                    return -1

                log_im = torch.log(torch.abs(im)+np.spacing(1))
                log_im = (log_im-min_)/(max_-min_)
                log_im = torch.clip(log_im,0,1)
                return log_im.float()
            else:
                print('Data type {} unknown, can not use denormalization function'.format(type(im)))
                return -1
               
class denormalization():
    def __init__(self,m,M) -> None:
        self.m_ = m
        self.M_ = M
    def __call__(self,im):
        if isinstance(im,np.ndarray):
            return (np.exp((self.M_ - self.m_) * (np.squeeze(im)).astype('float32') + self.m_)-np.spacing(1))
        elif torch.is_tensor(im):
            min_ = torch.from_numpy(self.m_)
            max_ = torch.from_numpy(self.M_)
            min_ = min_.to('cuda')
            max_ = max_.to('cuda')
            return (torch.exp((max_ - min_)*im + min_)-np.spacing(1))
        else:
            print('Data type {} unknown, can not use denormalization function'.format(type(im)))
            return -1

class denormalization_():
    def __call__(self,im,m,M):
        if isinstance(im,np.ndarray):
            return (np.exp((M-m) * (np.squeeze(im)).astype('float32') + m)-np.spacing(1))
        elif torch.is_tensor(im):
            if isinstance(m,np.ndarray) and isinstance(M,np.ndarray):
                min_ = torch.from_numpy(m).to(self.device)
                max_ = torch.from_numpy(M).to(self.device)
            elif isinstance(m,(float,int)) and isinstance(M,(float,int)):
                min_ = m
                max_ = M
            else :
                print("normalization parameter should be a float or an array")
                return -1
            return (torch.exp((max_ - min_)*im + min_)-np.spacing(1))
        else:
            print('Data type {} unknown, can not use denormalization function'.format(type(im)))
            return -1
        
class shift():
    def __init__(self,m,M,p) -> None:
        assert 0 <= p <= 1
        assert m <= M
        self.m = m # minimum shift value
        self.M = M # maximum shift value
        self.p = p # shift probablility
    def __call__(self, im):
        if np.random.uniform() < self.p:
            if isinstance(im,np.ndarray):
                return np.clip(im+np.random.uniform(self.m,self.M),0,1)
            elif torch.is_tensor(im):
                s = self.m + torch.rand(1,device=im.device) * (self.M-self.m)
                return torch.clip(im+s,0,1)
        return im
            
class sub_sample():
    def __init__(self,scale, center=False) -> None:
        self.center = center
        self.scale = scale

    def sub_sample_ch(self,im):
        dim = im.shape[0]
        ratio = dim/self.scale
        up = dim//2 - self.scale//2
        img_f = np.fft.fftshift(np.fft.fft2(im))
        img_f_s = img_f[up:up+self.scale,up:up+self.scale]
        img_f_s = np.fft.ifft2(np.fft.ifftshift(img_f_s / ratio**2))
        # print(np.mean(im))
        # print(np.mean(img_f_s))
        # disp_sar(im)
        # disp_sar(img_f_s)

        return img_f_s

    def __call__(self, im):

        channels = im.shape[2]
        for ch in range(channels):
            im_ch = im[:,:,ch]
            if self.center:
                im = symetrisation_patch(im_ch)
            im_ch = self.sub_sample_ch(im_ch)
            if ch == 0:
                sub_im = im_ch[:,:,np.newaxis]
            else:
                sub_im = np.concatenate((sub_im,im_ch[:,:,np.newaxis]),axis=2)
        return sub_im

class DataAugmentationSAR():
    def __init__(self, global_crops_scale, local_crops_scale, subres_crop_scale, global_crops_number, local_crops_number, shift_proba=0,shift_min=0, shift_max=0, subres_crop = False,device='cuda'):

        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.subres_crop = subres_crop
        # normalize the data for the specific sensor
        self.norm = normalization_(device=device)
        # global crop
        self.global_transfo = transforms.Compose([ToTensor(),transforms.RandomCrop(global_crops_scale,padding=None)])
        # small crops
        self.local_transfo = transforms.Compose([ToTensor(),transforms.RandomCrop(local_crops_scale,padding=None)])
        # sub resolution (no over sampling)
        self.subres_transfo = transforms.Compose([sub_sample(subres_crop_scale),ToTensor()])
        # reflectivity shift
        self.shift = shift(m = shift_min, M = shift_max, p = shift_proba)

    def __call__(self, slc, denoised, param_norm, param_subsample):
        crops = []
        # teacher with denoised data
        crops.append(self.norm(self.global_transfo(denoised),param_norm[0],param_norm[1]))
        # student with noisy data
        for _ in range(self.global_crops_number):
            crops.append(self.shift(self.norm(self.global_transfo(slc),param_norm[0],param_norm[1])))
        for _ in range(self.local_crops_number):
            crops.append(self.shift(self.norm(self.local_transfo(slc),param_norm[0],param_norm[1])))
        if self.subres_crop and param_subsample:
            crops.append(self.shift(self.norm(self.subres_transfo(slc),param_norm[0],param_norm[1])))
        return crops
    
class TransformEvalSar():
    def __init__(self, device='cuda') -> None:
        self.norm = normalization_(device=device)
        self.tens = ToTensor()
    def __call__(self, im, param_norm):
        return self.norm(self.tens(np.abs(im)),param_norm[0],param_norm[1])
    

class SegmentationAugmentation():
    def __init__(self,param_norm,train=True,p=0.5):
        self.transfo = transforms.Compose([normalization(param_norm[0],param_norm[1]),ToTensor()])
        self.train = train
        if train :
            self.p = p
            self.hflip = v2.RandomHorizontalFlip(p=1)
            self.vflip = v2.RandomVerticalFlip(p=1)
        
    def __call__(self, image, mask):

        image = self.transfo(image)
        if self.train:
            if np.random.uniform() < self.p:
                image = self.hflip(image)
                mask = self.hflip(mask)
            if np.random.uniform() < self.p:
                image = self.vflip(image)
                mask = self.vflip(mask)
        return image,mask
    
    
class amplitude_norm():
    def __init__(self,M=None):
        self.M = M

    def __call__(self,im):
        if self.M == None:
            M = np.mean(im)+3*np.std(im)
        else :
            M = self.M
        if isinstance(im,np.ndarray): 
            tensor = np.clip(np.abs(im)/M,0,1)
            tensor = (tensor-0.5)/0.5
            return tensor
        elif torch.is_tensor(im):
            tensor = torch.clip(torch.abs(im)/M,0,1)
            tensor = (tensor-0.5)/0.5
            return tensor