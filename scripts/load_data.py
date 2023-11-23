import glob
import tarfile
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class tarUnzipper:
    @staticmethod
    def get_data(dir, 
                 langs,
                 col_names=['index', 'original', 
                            'translation', 'scores',
                            'mean', 'z_scores', 
                            'z_mean', 'model_scores'],
                 drop_cols=['index'],
                 col_fn={'scores': json.loads,
                         'z_scores': json.loads,
                         'mean': np.float16,
                         'z_mean': np.float16,
                         'model_scores': np.float16,
                         'index': np.int16}):

        paths = [path for path in glob.glob(dir + '*' if dir.endswith('/') else dir + '/*')]

        col_names = col_names + ['original_lang'] if 'original_lang' not in col_names else col_names
        col_names = col_names + ['translation_lang'] if 'translation_lang' not in col_names else col_names
        
        df = pd.DataFrame(columns=col_names)

        data_dict = {col_name: list() for col_name in col_names}

        for p in paths:
            path, filename = '/'.join(p.split('/')[:-1]) + '/', p.split('/')[-1]
            src, tgt = filename.split('-')[:2]
            if all(lang in langs for lang in [src, tgt]):
                with tarfile.open(path + filename, 'r:gz') as f:
                    for member in f:
                        if member.name.endswith('.tsv'):
                            content = f.extractfile(member)
                            content = content.readlines()[1:]
                            for line in content:
                                line = line.decode().strip()
                                items = line.split('\t') + [src, tgt]
                                [data_dict[key].append(item) for key, item in zip(col_names, items)]

        df = pd.DataFrame(data_dict)
        
        for key, fn in col_fn.items():
            df[key] = df[key].apply(fn)
        
        if drop_cols:
            df = df.drop(columns=drop_cols)      
        
        return df
    
class QEDataset(Dataset):
    def __init__(self, 
                 path, 
                 langs, 
                 col_names=['index', 'original', 
                            'translation', 'scores',
                            'mean', 'z_scores', 
                            'z_mean', 'model_scores'], 
                 drop_cols=['index'],
                 col_fn={'scores':json.loads,
                         'z_scores':json.loads,
                         'mean': np.float16,
                         'z_mean': np.float16,
                         'model_scores': np.float16,
                         'index': np.int16}):
        super().__init__()
        self.data = self.__setup(path, 
                                 langs, 
                                 col_names=col_names, 
                                 drop_cols=drop_cols,
                                 col_fn=col_fn)

    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def __len__(self):
        return len(self.data.index)

    def __setup(self, 
                path, langs, 
                col_names, 
                drop_cols, 
                col_fn):
        return tarUnzipper.get_data(path, 
                                    langs, 
                                    col_names=col_names, 
                                    drop_cols=drop_cols, 
                                    col_fn=col_fn)
    
    @staticmethod
    def collate_fn(batch):
        return pd.DataFrame(batch)
        

if __name__ == '__main__':
    path = "/home/norrman/GitHub/RND/data/direct-assessments/train"
    langs = "en", "de", "ro" , "ru"
    
    dataset = QEDataset(path, langs)
    dataloader = iter(DataLoader(dataset, batch_size=120, 
                                 shuffle=True, 
                                 collate_fn=QEDataset.custom_collate_fn))

    batch = next(dataloader)
    

    