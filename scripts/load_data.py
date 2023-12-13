import glob
import tarfile
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import torch
import numpy
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize

from spacy.tokenizer import Tokenizer
from spacy.lang.ro import Romanian
from spacy.lang.en import English
from spacy.lang.ru import Russian
from spacy.lang.et import Estonian
from spacy.lang.zh import Chinese
from spacy.lang.de import German

import nltk

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

def get_dataset_stats(dataset):
    scores = numpy.array(dataset.data['mean'].tolist())
    orig_sent_len = list()
    trans_sent_len = list()
    orig_unique_toks = set()
    trans_unique_toks = set()

    for index, row in dataset.data.iterrows():
        if row['translation_lang'] == 'en':
            trans_lang = 'english'

        elif row['translation_lang'] == 'de':
            trans_lang = 'german'
         
        elif row['translation_lang'] == 'zh':
            trans_lang = 'chinese'
    
        if row['original_lang'] == 'ru':
            orig_lang = 'russian'
  
        elif row['original_lang'] == 'ro':
            orig_lang = 'romanian'
    
        elif row['original_lang'] == 'en':
            orig_lang = 'english'
   
        elif row['original_lang'] == 'et':
            orig_lang = 'estonian'


        orig_toks = word_tokenize(row['original'], language=orig_lang)


        orig_unique_toks.update(set(orig_toks))
        # trans_unique_toks.update(set(trans_toks))

        orig_sent_len.append(len(orig_toks))
        # trans_sent_len.append(len(trans_toks))

    return pd.DataFrame({'mean_score': scores.mean(), 
                            'std_score': scores.std(), 
                            'orig_avg_sent_len': numpy.array(orig_sent_len).mean(),
                            'orig_std_sent_len': numpy.array(orig_sent_len).std(),
                            # 'trans_avg_sent_len': numpy.array(trans_sent_len).mean(),
                            # 'trans_std_sent_len': numpy.array(trans_sent_len).std(),
                            'orig_total_toks': numpy.array(orig_sent_len).sum(),
                            'orig_unique_toks': len(orig_unique_toks),
                            # 'trans_total_toks': numpy.array(trans_sent_len).sum(),
                            # 'trans_unique_toks': len(trans_unique_toks)
                        }, index=[0])

def get_hundred(dataset):
    sample = dataset.data.sample(n=100)
    sample = sample[['original', 'translation']]

    return sample

if __name__ == '__main__':
    path = "/home/norrman/GitHub/RND/data/direct-assessments/test"
    langs = "en", "zh"
    
    dataset = QEDataset(path, langs)

    get_dataset_stats(dataset).to_csv('en_zh_test_dataset_stats.csv')
    
    

    