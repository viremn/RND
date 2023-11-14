import glob
import tarfile
import pandas as pd
from io import BytesIO

PATH = "/home/norrman/GitHub/RND/data/direct-assessments/"
LANGS = "en", "de", "zh", "ru", "ro", "et"
DATASET = 'dev'
COL_NAMES = ['index', 'original', 'translation', 'scores', 
             'mean', 'z_scores', 'z_mean', 'model_scores']

class tarUnzipper:
    @staticmethod
    def get_data(path, 
                 langs, 
                 datasplit,
                 drop_cols=None, 
                 col_names=['index', 'original', 
                            'translation', 'scores',
                            'mean', 'z_scores', 
                            'z_mean', 'model_scores']):
        paths = glob.glob(PATH + DATASET + '/*')
        files = dict()
        dfs = dict()

        for file in paths:
            path, filename = '/'.join(file.split('/')[:-1]) + '/', file.split('/')[-1]
            src, tgt = filename.split('-')[:2]
            files[(src, tgt)] = {'path': path, 'filename': filename}
        for key in files.keys():
            if all(lang in LANGS for lang in key):
                with tarfile.open(files[key]['path'] + files[key]['filename'], 'r:gz') as f:
                    for member in f:
                        if member.name.endswith('.tsv'):
                            content = f.extractfile(member)
                            dfs[key] = pd.read_csv(content, header=0, on_bad_lines='skip', sep='\t')
                            dfs[key].columns = col_names
                            if drop_cols:
                                dfs[key] = dfs[key].drop(columns=drop_cols)
                            dfs[key]['original_lang'] = key[0]
                            dfs[key]['translation_lang'] = key[1]

        return pd.concat(dfs.values(), axis=0, ignore_index=True)
    
        
            
        

if __name__ == '__main__':
    
    print(tarUnzipper.get_data(PATH, LANGS, DATASET, drop_cols=['index', 'z_mean', 'model_scores', 'z_scores']))