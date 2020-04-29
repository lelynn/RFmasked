import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as dataset
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.tensor as tensor
from tqdm import tqdm
import seaborn as sns

import model_file
import module_split as module
import RF_module as RF
import scipy.io as sio
import os

import sys
sys.path.append('')

# -----
# Some params
# -----
runname = '191_splARGERnewtv_wnograd'
device = 0
cuda0 = torch.device(f'cuda:{device}')
batch_size = 4
epochs = 200

in_channels=191

if in_channels == 2:
    inputtype = 'V1_V4'
if in_channels == 191:
    inputtype = 'all_channels'
 
# -----
# Model, loss, & optimizer
# -----
model = model_file.ResblocksDeconv(in_channels, (240,240))


if __name__ == '__main__':
    
    if device >= 0:
        model.cuda(device)
        
#     lossFunction = module.LossFunction(device)
    lossFunction = module.VGGLoss(device)
    optimizer = optim.Adam(model.parameters(), )

    hori_means, verti_means, std_avg = RF.extract_means_std()
    # -----
    # RF gaus maps
    # ------
    gaus = module.load_gausdata()
    seen_images = module.load_ydata()

    # ------
    # Training
    # ------
    nn_training = module.load_LFPdata('training')
    training_iterator = module.make_iterator(nn_training, 'training', batch_size, shuffle = True)

    # ------
    # Testing
    # ------
    nn_testing = module.load_LFPdata('testing')
    # testing_targets = module.load_ydata('testing')
    testing_iterator = module.make_iterator(nn_testing, 'testing', batch_size, shuffle = False)

    # -----
    # EPOCHS
    # -----

    losses_train = []
    losses_test = []
    confidence_mask = RF.make_confidence_mask(hori_means, verti_means, std_avg)
    confidence_mask = torch.from_numpy(confidence_mask.astype('float32')).to(cuda0)

    for e in range(epochs):  # loop over the dataset multiple times
        loss_train = 0
        model.train()
                               
        for sample, target_indices in tqdm(training_iterator, total=len(training_iterator)):
            # -----
            # Inputs
            # -----
            mean_signals  = sample[:,300]
            gaus_expand_to_batch = gaus.expand([len(target_indices), 191, 240, 240])
            mean_signals_expand_to_gaus = mean_signals.expand([240, 240, len(target_indices), 191])
            mean_signals_expand_to_gaus = mean_signals_expand_to_gaus.transpose(0,2).transpose(1,3)
            inputs = module.select_type_inputs(inputtype, gaus_expand_to_batch, mean_signals_expand_to_gaus)
            inputs = inputs.to(cuda0)

            # -----
            # Targets
            # -----
            target_batch = seen_images[target_indices]
            target_batch = target_batch.transpose(3,1).transpose(2,3)
            target_batch = target_batch.to(cuda0)

            # -----
            # Outputs
            # -----
            optimizer.zero_grad()
            y = model(inputs) 

            # -----
            # Before calculating loss, make a mask
            # -----
            y *= confidence_mask.expand_as(y)
            target_batch *= confidence_mask.expand_as(target_batch)

            # ------
            # Loss 
            # ------
            train_loss = lossFunction(y, target_batch)

            # ------
            # Backward & update
            # ------
            train_loss.backward()
            optimizer.step()

            # ------
            # Loss 
            # ------
            loss_train += train_loss.sum().item()
        losses_train.append(loss_train/len(training_iterator.sampler))
    
        with torch.no_grad():
            loss_test = 0
            model.eval()
            for sample, target_indices in tqdm(testing_iterator, total=len(testing_iterator)):
                # -----
                # Inputs
                # -----
                mean_signals  = sample[:,300]
                gaus_expand_to_batch = gaus.expand([len(target_indices), 191, 240, 240])
                mean_signals_expand_to_gaus = mean_signals.expand([240, 240, len(target_indices), 191])
                mean_signals_expand_to_gaus = mean_signals_expand_to_gaus.transpose(0,2).transpose(1,3)

                inputs = module.select_type_inputs(inputtype, gaus_expand_to_batch, mean_signals_expand_to_gaus)
                inputs = inputs.to(cuda0)
                # -----
                # Targets
                # -----
                target_batch = seen_images[target_indices]
                target_batch = target_batch.transpose(3,1).transpose(2,3)
                target_batch = target_batch.to(cuda0)
                # -----
                # Outputs
                # -----
                y = model(inputs)
                y *= confidence_mask.expand_as(y)
                target_batch *= confidence_mask.expand_as(target_batch)
                # ------
                # Loss 
                # ------
                test_loss = lossFunction(y, target_batch)
                loss_test += test_loss.sum().item()

        losses_test.append(loss_test/len(testing_iterator.sampler))

        os.makedirs(runname, exist_ok=True)
        np.save(f'{runname}/loss_train', np.array(losses_train))
        np.save(f'{runname}/loss_test', np.array(losses_test))
        torch.save(model.state_dict(), f'{runname}/epochs_{e}.model')
        print('epochs: ', e)
    os.makedirs(runname, exist_ok=True)
    np.save(f'{runname}/loss_train', np.array(losses_train))
    np.save(f'{runname}/loss_test', np.array(losses_test))
    torch.save(model.state_dict(), f'{runname}/epochs_final.model')
