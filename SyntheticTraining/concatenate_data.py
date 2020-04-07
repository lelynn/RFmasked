import synthetic_fun
from tqdm import tqdm
import numpy as np

# synthetic_fun.make_unique_synth('training')

all_batches = []
set_t = 'training'
for batch_i in tqdm(range(8)):
    batch = np.load(f'../../sorted_data/LFP/{set_t}/{set_t}_synth191batch{batch_i}.npy')
    all_batches.append(batch)
    inputs191_list = np.concatenate(all_batches, axis=0)
    np.save(f'../../sorted_data/LFP/{set_t}/{set_t}_synth191final.npy', inputs191_list.astype('float32'))

    
set_t = 'testing'

batch = np.load(f'../../sorted_data/LFP/{set_t}/{set_t}_synth191batch0.npy')
all_batches.append(batch)
inputs191_list = np.concatenate(all_batches, axis=0)
np.save(f'../../sorted_data/LFP/{set_t}/{set_t}_final.npy', inputs191_list.astype('float32'))