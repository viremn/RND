import glob
import torch
import pandas as pd

from load_data import QEDataset
from qemodel import QEModel, eval_model, predict

from multilingual_embeddings import MultilingualStaticSentenceEmbedder, \
    LASERSentenceEmbedder, DistilUSEEmbedder, ParaphraseEmbedder, \
    LaBSEEmbedder, BertSentenceEmbedder, XLMREmbedder

# yield 'static', MultilingualStaticSentenceEmbedder(embedding_file_path=static_emb_path,
#                                              langs=langs)
#     yield 'laser', LASERSentenceEmbedder()
#     yield 'distil', DistilUSEEmbedder()
#     yield 'paraphrase', ParaphraseEmbedder()
#     yield 'labse', LaBSEEmbedder()
#     yield 'bert-cls', BertSentenceEmbedder(pooling='cls')
#     yield 'bert-mean', BertSentenceEmbedder(pooling='mean')
#     yield 'xlmr', XLMREmbedder()

emb_type = 'xlmr'

path = f'/media/norrman/Expansion Drive/Git_Projects/RND/tailored_parameter_models/{emb_type}_best_params/'
outpath = f'/home/norrman/GitHub/RND/models/tailored_parameter_models/{emb_type}_best_params/run1/'

last_checkpoints = [item for item in glob.glob(path+'*') if item.endswith('last.checkpoint.pt')]
best_checkpoints = [item for item in glob.glob(path+'*') if item.endswith('best.checkpoint.pt')]

checkpoints = [(cb, cl) for cb in best_checkpoints for cl in last_checkpoints if cb.split('.')[0] == cl.split('.')[0]]

settings = {'Learning Rate': 0.00023658986073689422,
                'Weight Decay': 1.00E-05,
                'Dropout': 0.1,
                'LR Scheduler Factor': 0.1,
                'Encoder Depth': 2,
                'Encoder Size': 1024,
                'Estimator Depth': 1,
                'Estimator Size': 4096,
                'Shared-Weights': True}

en_zh_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                langs=['en', 'zh'])
et_en_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                langs=['en', 'et'])

for best, last in checkpoints:
    load_best = torch.load(best)
    load_last = torch.load(last)

    outname = best.split("/")[-1].split(".")[0]

    print('evaluating:', outname)

    if 'static' in best.split("/")[-1].split('_')[0]:
        continue
    elif 'laser' in best.split("/")[-1].split('_')[0]:
        embedder = LASERSentenceEmbedder()
    elif 'distil' in best.split("/")[-1].split('_')[0]:
        embedder = DistilUSEEmbedder()
    elif 'paraphrase' in best.split("/")[-1].split('_')[0]:
        embedder = ParaphraseEmbedder()
    elif 'labse' in best.split("/")[-1].split('_')[0]:
        embedder = LaBSEEmbedder() 
    elif 'bert-cls' in best.split("/")[-1].split('_')[0]:
        embedder = BertSentenceEmbedder(pooling='cls')
    elif 'bert-mean' in best.split("/")[-1].split('_')[0]:
        embedder = BertSentenceEmbedder(pooling='mean')
    elif 'xlmr' in best.split("/")[-1].split('_')[0]:
        embedder = XLMREmbedder()


    model = QEModel(embedder=embedder,
                    encoder_dim=settings['Encoder Size'],
                    encoder_depth=settings['Encoder Depth'],
                    shared_encoder_weights=settings['Shared-Weights'],
                    estimator_hidden_size=settings['Estimator Size'],
                    estimator_hidden_layers=settings['Estimator Depth'],
                    dropout=settings['Dropout'])
    
    model.load_state_dict(load_best['model_state_dict'])

    print('evaluating zh test')
    en_zh_eval = eval_model(model, en_zh_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(en_zh_eval, index=[0])
    eval_df.to_csv(f'{outpath+"zero_shot/"+outname}_best_checkpoint_en_zh_eval.csv')

    print('evaluating et test')  
    et_en_eval = eval_model(model, et_en_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(et_en_eval, index=[0])
    eval_df.to_csv(f'{outpath+"zero_shot/"+outname}_best_checkpoint_et_en_eval.csv')


    model = QEModel(embedder=embedder,
                    encoder_dim=settings['Encoder Size'],
                    encoder_depth=settings['Encoder Depth'],
                    shared_encoder_weights=settings['Shared-Weights'],
                    estimator_hidden_size=settings['Estimator Size'],
                    estimator_hidden_layers=settings['Estimator Depth'],
                    dropout=settings['Dropout'])
    
    model.load_state_dict(load_last['model_state_dict'])

    print('evaluating zh test')
    en_zh_eval = eval_model(model, en_zh_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(en_zh_eval, index=[0])
    eval_df.to_csv(f'{outpath+"zero_shot/"+outname}_last_checkpoint_en_zh_eval.csv')

    print('evaluating et test')  
    et_en_eval = eval_model(model, et_en_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(et_en_eval, index=[0])
    eval_df.to_csv(f'{outpath+"zero_shot/"+outname}_last_checkpoint_et_en_eval.csv')