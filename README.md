# RFmasked

Predicting masked areas of seen images based on the RF gaussians * brain signals. In this version, the provided test data of the dataset was not used. The privided training set was split using `splitting_data.ipynb` to create training and test sets. That is because the test set contained many repeats as shown in the below graph:
![/img_count.png]


<b> Gaussian RFs </b> are made (& visualized) in the notebook `multivariate_gaussian.ipynb`. Then the images were cropped using `cropping_images.ipynb`. 

The <b>training loop </b>can be found in `RFconfidence_mask_split.py`:

- Inputs are made by multiplying the brain signal (at t=300) with the gaussian RFs.
- Target is made using a confidence mask.
- Loss is calculated. 

The <b>models</b> used to train can be found in the `model_file_old.py` and `model_file.py`:

- `model_file_old.py` is the original model with nn.ConvTranspose2d().

- `model_file.py` contains unpooling layers instead and is a larger model. It's inspired by the DeconvNet from this [project](https://github.com/HyeonwooNoh/DeconvNet/tree/master/model).

`module_split.py` contains the loss and the dataloaders.
`RF_modules.py` contains the functions used for making the RF gaussians. 

`load_model_2.ipynb` loads the trained model with 2 channels and visualizes predictions. \
`load_model_191.ipynb` loads the trained model with 191 channels and visualizes predictions.

`train_on_one.ipynb` shows the model which is trained on one image (to induce overfitting). 
