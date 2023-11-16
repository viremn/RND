import glob
import tarfile
import pandas as pd

class tarUnzipper:
    @staticmethod
    def get_data(dir, 
                 langs,
                 drop_cols=None, 
                 col_names=['index', 'original', 
                            'translation', 'scores',
                            'mean', 'z_scores', 
                            'z_mean', 'model_scores']):

        paths = [path for path in glob.glob(dir + '*' if dir.endswith('/') else dir + '/*')]


        dfs = dict()

        for p in paths:
            path, filename = '/'.join(p.split('/')[:-1]) + '/', p.split('/')[-1]
            src, tgt = filename.split('-')[:2]
            key = src, tgt
            if all(lang in langs for lang in key):
                with tarfile.open(path + filename, 'r:gz') as f:
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
    path = "/home/norrman/GitHub/RND/data/direct-assessments/train"
    langs = "en", "de"
    col_names = ['index', 'original', 'translation', 'scores', 
                'mean', 'z_scores', 'z_mean', 'model_scores']

    print(tarUnzipper.get_data(path, langs, drop_cols=['index', 'z_mean', 'model_scores', 'z_scores']))