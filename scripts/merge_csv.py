import glob
import pandas as pd

path = '/home/norrman/GitHub/RND/models/uniform_model_settings/model_evaluations'

files = glob.glob(path+'/*')

combined_csv = pd.DataFrame(columns=['total_loss', 'avg_loss', 'correlation', 'p'])

for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
        name = file.split('/')[-1].split('_')[0]\
                +'_'+('best' if 'best' in file else 'last')\
                +'_'+('mixed' if 'mixed' in file.split('/')[-1].split('_')[-3:] 
                      else ('ru' if 'ru' in file.split('/')[-1].split('_')[-3:]
                            else ('ro' if 'ro' in file.split('/')[-1].split('_')[-3:] 
                                  else 'de')))
        items = lines[1].strip().split(',')[1:]
        combined_csv.loc[name] = items

combined_csv.to_csv('combined.csv')