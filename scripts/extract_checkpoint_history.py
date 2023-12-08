import glob
import torch
import pandas as pd

path = '/home/norrman/GitHub/RND/models/tailored_parameter_models/laser_best_params/run2/'

checkpoints = [item for item in glob.glob(path+'*') if item.endswith('last.checkpoint.pt')]
print(checkpoints)

for checkpoint in checkpoints:
    loaded_checkpoint = torch.load(checkpoint)
    history = loaded_checkpoint['history']
    df = pd.DataFrame(history)
    df.to_csv(path+checkpoint.split('/')[-1].split('.')[0]+'.history.csv')