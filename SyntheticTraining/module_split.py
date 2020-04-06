#################
# IMPORTS
#################
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import torch.utils.data as dataset
import sys
from torch.autograd import Variable

sys.path.append('')
#################
# CLASSES
#################

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            with torch.no_grad():
                for param in self.parameters():
                    param = Variable(param, volatile=True)
                    param.requires_grad = False

    def forward(self, X):
#         X = Variable(X, volatile=True)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        if gpu_ids >= 0:
            self.vgg = Vgg19().cuda(device = gpu_ids)
        else:
            self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.mean_sq0 = nn.MSELoss()
        self.mean_sq1 = nn.MSELoss()

    def forward(self, x, y, pix = 0.1):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        pixel_loss = pix * self.mean_sq1(x, y)
        
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())  
        loss = loss + pixel_loss
        return loss
    
    
# class TotalVariationLoss(object):
#     def __init__(self, device):
        
#         if device >= 0:
#             self.cuda0 = torch.device(f'cuda:{device}')
#         else:
#             self.cuda0 = 'cpu'

#     def tv_loss(self, x):
#         h1 = torch.sum((x[:,:,:,1:]-x[:,:,:,:-1])**2)
#         h2 = torch.sum((x[:,:,1:,:]-x[:,:,:-1,:])**2)

#         y = h1 + h2
        
#         if self.cuda0 != 'cpu':
#             y = y.to(self.cuda0)
        
#         return y 


#class LossFunction(object):
#    def __init__(self, device):
#        if device >= 0:
#            cuda0 = torch.device(f'cuda:{device}')
#            vgg = models.vgg16(pretrained=True).to(cuda0)
#            self.totalVariationLoss = TotalVariationLoss(device)
#        else:
#            vgg = models.vgg16(pretrained=True)
#            self.totalVariationLoss = TotalVariationLoss(device)
#        
#        self.mean_sq0 = nn.MSELoss()
#        self.mean_sq1 = nn.MSELoss()
#
#        self.vgg16Layers = nn.Sequential(*list(vgg.children())[0])[:11]
#        
#        for param in self.vgg16Layers.parameters():
#            param.requires_grad = False
#
#    def __call__(self, t, y, pix = 0.5, tv=1e-8):
#        
#        t_ = self.vgg16Layers(t)
#        y_ = self.vgg16Layers(y)
#        feature = self.mean_sq0(t_, y_)
#        pixel_loss = pix * self.mean_sq1(t, y)
#
#        total_variation_loss = tv * self.totalVariationLoss.tv_loss(y)
##         loss = feature + pixel_loss + total_variation_loss
#        loss = feature + pixel_loss
#
#        return loss
    
#################
# FUNCTIONS
#################


def load_gausdata():
    
    ''' 
    
    Load all necessary items for creating inputs for the model. 
    Inputs for the model will be signals * multivariate gaussian RFs. T
    he targets will be just the images seen by the monkey. returns gaus
    '''
    
    # Receptive_field fitted with multivariate function (2D)
    gaus_raw = np.load('multivariate_gaussians240_cropped.npy') # (192 x 240 x 240) is the shape
    gaus = np.concatenate([gaus_raw[:144], gaus_raw[145:]]) #takes away the ones with respective Nan electrode (elec # = 144) in LFP data
    gaus = torch.from_numpy(gaus).float()
    return gaus

def load_LFPdata(set_t):
    
    '''
    
    Loads the LFP data.
    -------------------
    set_t: 'training' for loading training data, 'testing' for testing data.
    -------------------
    
    Returns float32. 
    '''
    
    nn_data = np.load(f'../../sorted_data/LFP/{set_t}/LFP_{set_t}_splitted.npy')
    return nn_data.astype('float32')

def load_ydata():
    '''Loads targets.
    
    ------------------
    These are himages from the old testing dataset before splitting. Now since the data is slit, use only this one and not testing
    cropped images. 
    Returns a torch.Tensor()'''

#     targets = np.load(f'splitcrop_{set_t}_im.npy')
    targets = np.load(f'cropped_images_training.npy')

    return targets

  
# def make_iterator(nn_data, img_indexes, set_t, batch_size, shuffle):
#     '''
    
#     Makes an iterator for this experiment. It allows iteration through indices and the signals
    
#     Returns an iterator.'''
# #     img_indexes = np.load(f'../../sorted_data/LFP/{set_t}/index_{set_t}_LFP.npy').astype(np.int)
#     data_indices = torch.from_numpy(img_indexes)
#     data = dataset.TensorDataset(torch.from_numpy(nn_data.T), data_indices)
    
#     return dataset.DataLoader(data, batch_size, shuffle = shuffle)

def make_iterator(nn_data, set_t, batch_size, shuffle):
    '''
    
    Makes an iterator for this experiment. It allows iteration through indices and the signals
    
    Returns an iterator.'''
    
    
    img_indexes = np.load(f'../../sorted_data/LFP/{set_t}/index_{set_t}_LFP_split.npy').astype(np.int)
    data_indices = torch.from_numpy(img_indexes)
    data = dataset.TensorDataset(torch.from_numpy(nn_data.T), data_indices)
    
    return dataset.DataLoader(data, batch_size, shuffle = shuffle)

def select_type_inputs(type_input, gaus_expand_to_batch, mean_signals_expand_to_gaus):
    '''
    
    Selects which type of inputs will be used for the model.
    If 'all_channels', set to in channels as 191. If 'V1_V4', set
    in_channels as 2.
    
    '''
    if type_input == 'all_channels':        
        return (gaus_expand_to_batch * mean_signals_expand_to_gaus)
    
    if type_input == 'V1_V4':
        weighted_gaus = gaus_expand_to_batch * mean_signals_expand_to_gaus
        v1 = weighted_gaus[:,:97].sum(1)
        v4 = weighted_gaus[:,97:].sum(1)

        return torch.stack((v1,v4), dim=1)
