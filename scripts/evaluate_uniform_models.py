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

path = '/home/norrman/GitHub/RND/models/uniform_model_settings/'

last_checkpoints = [item for item in glob.glob(path+'*') if item.endswith('last.checkpoint.pt')]
best_checkpoints = [item for item in glob.glob(path+'*') if item.endswith('best.checkpoint.pt')]

checkpoints = [(cb, cl) for cb in best_checkpoints for cl in last_checkpoints if cb.split('.')[0] == cl.split('.')[0]]

settings = {'Learning Rate': 1.00E-03,
                'Weight Decay': 1.00E-05,
                'Dropout': 0.2,
                'LR Scheduler Factor': 0.1,
                'Encoder Depth': 2,
                'Encoder Size': 1024,
                'Estimator Depth': 1,
                'Estimator Size': 4096,
                'Shared-Weights': True}

mixed_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                langs=['en', 'de', 'ru', 'ro'])
en_de_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                langs=['en', 'de'])
ru_en_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                langs=['en', 'ru'])
ro_en_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                langs=['en', 'ro'])

for best, last in checkpoints:
    load_best = torch.load(best)
    load_last = torch.load(last)

    outname = best.split("/")[-1].split(".")[0]

    print('evaluating:', outname)

    if 'static' in best:
        embedder = MultilingualStaticSentenceEmbedder(embedding_file_path='/home/norrman/GitHub/RND/data/embeddings/multilingual_embeddings.tar.gz',
                                                      langs=['en', 'de', 'ro', 'ru'])
    elif 'laser' in best:
        embedder = LASERSentenceEmbedder()
    elif 'distil' in best:
        embedder = DistilUSEEmbedder()
    elif 'paraphrase' in best:
        embedder = ParaphraseEmbedder()
    elif 'labse' in best:
        embedder = LaBSEEmbedder() 
    elif 'bert-cls' in best:
        embedder = BertSentenceEmbedder(pooling='cls')
    elif 'bert-mean' in best:
        embedder = BertSentenceEmbedder(pooling='mean')
    elif 'xlmr' in best:
        embedder = XLMREmbedder()


    model = QEModel(embedder=embedder,
                    encoder_dim=settings['Encoder Size'],
                    encoder_depth=settings['Encoder Depth'],
                    shared_encoder_weights=settings['Shared-Weights'],
                    estimator_hidden_size=settings['Estimator Size'],
                    estimator_hidden_layers=settings['Estimator Depth'],
                    dropout=settings['Dropout'])
    
    model.load_state_dict(load_best['model_state_dict'])

    print('evaluating mixed test')
    mixed_eval = eval_model(model, mixed_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(mixed_eval, index=[0])
    eval_df.to_csv(f'{outname}_best_checkpoint_mixed_eval.csv')

    print('evaluating ru test')  
    ru_en_eval = eval_model(model, ru_en_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(ru_en_eval, index=[0])
    eval_df.to_csv(f'{outname}_best_checkpoint_ru_en_eval.csv')

    print('evaluating ro test')
    ro_en_eval = eval_model(model, ro_en_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(ro_en_eval, index=[0])
    eval_df.to_csv(f'{outname}_best_checkpoint_ro_en_eval.csv')

    print('evaluating de test')
    en_de_eval = eval_model(model, en_de_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(en_de_eval, index=[0])
    eval_df.to_csv(f'{outname}_best_checkpoint_en_de_eval.csv')

    model = QEModel(embedder=embedder,
                    encoder_dim=settings['Encoder Size'],
                    encoder_depth=settings['Encoder Depth'],
                    shared_encoder_weights=settings['Shared-Weights'],
                    estimator_hidden_size=settings['Estimator Size'],
                    estimator_hidden_layers=settings['Estimator Depth'],
                    dropout=settings['Dropout'])
    
    model.load_state_dict(load_last['model_state_dict'])

    print('evaluating mixed test')
    mixed_eval = eval_model(model, mixed_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(mixed_eval, index=[0])
    eval_df.to_csv(f'{outname}_last_checkpoint_mixed_eval.csv')

    print('evaluating ru test')  
    ru_en_eval = eval_model(model, ru_en_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(ru_en_eval, index=[0])
    eval_df.to_csv(f'{outname}_last_checkpoint_ru_en_eval.csv')

    print('evaluating ro test')
    ro_en_eval = eval_model(model, ro_en_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(ro_en_eval, index=[0])
    eval_df.to_csv(f'{outname}_last_checkpoint_ro_en_eval.csv')

    print('evaluating de test')
    en_de_eval = eval_model(model, en_de_test_dataset, batch_size=32)
    eval_df = pd.DataFrame(en_de_eval, index=[0])
    eval_df.to_csv(f'{outname}_last_checkpoint_en_de_eval.csv')