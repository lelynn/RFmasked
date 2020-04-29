import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as dataset
from tqdm import tqdm

import model_file
import module_split as module
import RF_module as RF

import os

runname = 'synthetic_loaddot191_Adam1'
device = 1
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
    # lossFunction = module.LossFunction(device)
    lossFunction = module.VGGLoss(device)
    optimizer = optim.Adam(model.parameters(), 0.1)

    hori_means, verti_means, std_avg = RF.extract_means_std()

    # -----
    # Inputs:
    # Will be dot number times the gaus
    # ------
    gaus = module.load_gausdata()

    # ------
    # Targets:
    # are the masked nn_seen_torch, correspponding to the dot_number.
    # ------
    nn_seen_torch = torch.from_numpy(module.load_ydata())

    # ------
    # Training
    # ------
    dot_numbers_train = np.load(f'training/training_synth191final.npy')
    training_iterator = module.make_iterator_unique(dot_numbers_train, 'training', batch_size, shuffle = True)
    
    # ------
    # Testing
    # ------
    dot_numbers_test = np.load(f'testing/testing_synth191final.npy')
    testing_iterator = module.make_iterator_unique(dot_numbers_test, 'testing', batch_size, shuffle = False)

    # EPOCHS
    losses_train = []
    losses_test = []
    confidence_mask = RF.make_confidence_mask(hori_means, verti_means, std_avg)
    confidence_mask = torch.from_numpy(confidence_mask.astype('float32')).to(cuda0)
    for e in range(epochs):  # loop over the dataset multiple times
        loss_train = 0
        model.train()

        for dot_number, img_indices  in tqdm(training_iterator, total=len(training_iterator)):
            # -----
            # Inputs
            # -----

            gaus_expand_to_batch = gaus.expand([len(img_indices), 191, 240, 240])
            weight_images = dot_number[:,:,np.newaxis, np.newaxis].expand([len(img_indices), 191, 240, 240])     

            # We want to use the dot number and repeat it (expand to gaus) such that it will have the same shape. 
            #Then you multiply with the gaus_exapnd_go_batch!
            inputs = module.select_type_inputs(inputtype, gaus_expand_to_batch, weight_images)
            inputs = inputs.to(cuda0)

            # -----
            # Targets
            # -----
            target_batch = nn_seen_torch[img_indices]
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
            for dot_number, img_indices in tqdm(testing_iterator, total=len(testing_iterator)):
                # -----
                # Inputs
                # -----
                gaus_expand_to_batch = gaus.expand([len(img_indices), 191, 240, 240])
                weight_images = dot_number[:,:,np.newaxis, np.newaxis].expand([len(img_indices), 191, 240, 240])     


                # We want to use the dot number and repeat it (expand to gaus) such that it will have the same shape. 
                #Then you multiply with the gaus_exapnd_go_batch!
                inputs = module.select_type_inputs(inputtype, gaus_expand_to_batch, weight_images)
                inputs = inputs.to(cuda0)

                # -----
                # Targets
                # -----
                target_batch = nn_seen_torch[img_indices]
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
