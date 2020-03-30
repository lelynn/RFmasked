# RFmasked
A model trained to predict specific areas of seen images based on the RF gaussians * brain signals.

The training loop can be found in RFconfidence_mask_split.py. Here i also made the loss loops. Here the brain signals are multiplied by the gaussian RFs. Gaussian RFs are made (& visualized) in the notebook 'multivariate_gaussian.ipynb'.

The <b>models</b> used to train can be found in the `model_file_old.py` and `model_file.py`. 

'module_split.py' contains the loss and the dataloaders.

'RF_modules.py' contains the functions used for making the RF gaussians. 



