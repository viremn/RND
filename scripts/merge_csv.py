import glob
import pandas as pd

path = '/home/norrman/GitHub/RND/models/tailored_parameter_models/xlmr_best_params/run1/zero_shot/'

files = glob.glob(path+'*')

# combined_csv = pd.DataFrame(columns=['dropout','learning_rate','weight_decay','lr_scheduler_gamma','validation_accuracy','model_architecture'])
combined_csv = pd.DataFrame(columns=['total_loss','avg_loss','correlation','p'])


for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
        name = file.split('/')[-1].split('_')[0]\
                +'_'+('best' if 'best_checkpoint' in file else 'last')\
                +'_'+('mixed' if 'mixed' in file.split('/')[-1].split('_')[-3:] 
                    else ('ru' if 'ru' in file.split('/')[-1].split('_')[-3:]
                            else ('ro' if 'ro' in file.split('/')[-1].split('_')[-3:] 
                                else ('de' if 'de' in file.split('/')[-1].split('_')[-3:]
                                      else ('zh' if 'zh' in file.split('/')[-1].split('_')[-3:]
                                            else 'et')))))
        # name = file.split('/')[-1].split('.')[0]
        items = lines[1].strip().split(',')[1:]
        combined_csv.loc[name] = items

combined_csv.to_csv(path+'combined.csv')