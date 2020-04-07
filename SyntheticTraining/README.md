# Synthetic training

### Required data: 

RFfit.mat: This is the receptive field data. This data is used to make the <b> Gaussian RFs </b> in the notebook `multivariate_gaussian.ipynb`. 

cropped_images_tosplit.npy: are the <b> Seen Images</b> and is being used in the codes to split into training/testing data right before training (functions are defined in `module_split.py` and used in `train_synthetic_signals191`.)


### Generating synthetic data:
We make synthetic data using the function defined in `synthetic_fun.py`, which takes in the  <b> Gaussian RFs </b> and the <b> Seen Images</b>:

- <b> Seen Images</b> has the dimension [240, 240, 3]. We expand the <b> Seen Images</b> to have dimensions [240, 240, 3, 191] (191 being the number of receptive fields). This is then transposed and summed over color dimension now giving a shape of [191, 240, 240].

- <b> Gaussian RFs </b> has the dimension [191, 240, 240]. This is multiplied with the <b> Seen Images</b>, and then summed over 1st and 2nd dimension. ( <b> Gaussian RFs </b> * <b> Seen Images </b> ).sum((1,2)), leaving a dot product for each electrode, named <b> Dot Number </b>. This <b> Dot Number </b> is appended for every electrode, then stacked into numpy and saved as `{set_t}/{set_t}_synth191batch{batch_i}` {set_t} being `training` or `testing` and {batch_i} being the nth slice (since this is done in slices of 1000). Afterwards all data is appended and concatenated in `concatenate_data.py`. 



### Using synthetic data:\
In the training loop, the dot number is loaded along with its corresponding image index. The <b> Dot Number </b> is expanded to the same dimension as the <b> Gaussian RFs </b>. Inputs are calculated dependng on how many channels we want:
- 191 channels for separating the RFs:\
    Input191 = <b> Gaussian RFs </b> * <b> Expanded Dot Number </b> 
    
- 2 channels for separating V1 and V4 layers:\
        v1 = Input191[:,:97].sum(1)\
        v4 = Input191[:,97:].sum(1)\
        Input2 = torch.stack((v1,v4), dim=1)




### Other changes from the real data experiment

- data paths

- iterator: is made with the function: make_iterator_unique(), which takes unique image indices, since we use images as data and repeating images are not needed. 