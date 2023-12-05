from qemodel import QEModel, train_model, eval_model, predict, cross_fold_splits, hyper_parameter_search

from multilingual_embeddings import MultilingualStaticSentenceEmbedder, \
    LASERSentenceEmbedder, DistilUSEEmbedder, ParaphraseEmbedder, \
    LaBSEEmbedder, BertSentenceEmbedder, XLMREmbedder

from load_data import QEDataset

from torch.utils.data import Subset

import pandas as pd
import torch

def embedder_iter(static_emb_path, langs):
    # yield 'bert-max', BertSentenceEmbedder(pooling='max')
    yield 'static', MultilingualStaticSentenceEmbedder(embedding_file_path=static_emb_path,
                                                        langs=langs), {'Learning Rate': 1.00E-03,
                                                                        'Weight Decay': 1.00E-05,
                                                                        'Dropout': 0.2,
                                                                        'LR Scheduler Factor': 0.1,
                                                                        'Encoder Depth': 2,
                                                                        'Encoder Size': 1024,
                                                                        'Estimator Depth': 1,
                                                                        'Estimator Size': 4096,
                                                                        'Shared-Weights': True}
    yield 'laser', LASERSentenceEmbedder(), {'Learning Rate': 1.00E-03,
                                            'Weight Decay': 1.00E-05,
                                            'Dropout': 0.2,
                                            'LR Scheduler Factor': 0.1,
                                            'Encoder Depth': 2,
                                            'Encoder Size': 1024,
                                            'Estimator Depth': 2,
                                            'Estimator Size': 4096,
                                            'Shared-Weights': True}
    yield 'distil', DistilUSEEmbedder(), {'Learning Rate': 1.00E-03,
                                            'Weight Decay': 1.00E-05,
                                            'Dropout': 0.2,
                                            'LR Scheduler Factor': 0.1,
                                            'Encoder Depth': 1,
                                            'Encoder Size': 1024,
                                            'Estimator Depth': 1,
                                            'Estimator Size': 4096,
                                            'Shared-Weights': True}
    yield 'paraphrase', ParaphraseEmbedder(), {'Learning Rate': 1.00E-03,
                                                'Weight Decay': 1.00E-05,
                                                'Dropout': 0.2,
                                                'LR Scheduler Factor': 0.1,
                                                'Encoder Depth': 2,
                                                'Encoder Size': 1024,
                                                'Estimator Depth': 1,
                                                'Estimator Size': 4096,
                                                'Shared-Weights': True}
    yield 'labse', LaBSEEmbedder(), {'Learning Rate': 1.00E-03,
                                    'Weight Decay': 1.00E-05,
                                    'Dropout': 0.2,
                                    'LR Scheduler Factor': 0.1,
                                    'Encoder Depth': 1,
                                    'Encoder Size': 1024,
                                    'Estimator Depth': 2,
                                    'Estimator Size': 4096,
                                    'Shared-Weights': True}
    yield 'bert-cls', BertSentenceEmbedder(pooling='cls'), {'Learning Rate': 1.00E-03,
                                                            'Weight Decay': 1.00E-05,
                                                            'Dropout': 0.2,
                                                            'LR Scheduler Factor': 0.1,
                                                            'Encoder Depth': 2,
                                                            'Encoder Size': 1024,
                                                            'Estimator Depth': 1,
                                                            'Estimator Size': 4096,
                                                            'Shared-Weights': True}
    yield 'bert-mean', BertSentenceEmbedder(pooling='mean'), {'Learning Rate': 1.00E-03,
                                                            'Weight Decay': 1.00E-05,
                                                            'Dropout': 0.2,
                                                            'LR Scheduler Factor': 0.1,
                                                            'Encoder Depth': 1,
                                                            'Encoder Size': 1024,
                                                            'Estimator Depth': 1,
                                                            'Estimator Size': 4096,
                                                            'Shared-Weights': True}

    yield 'xlmr', XLMREmbedder(), {'Learning Rate': 1.00E-03,
                                    'Weight Decay': 1.00E-05,
                                    'Dropout': 0.2,
                                    'LR Scheduler Factor': 0.1,
                                    'Encoder Depth': 2,
                                    'Encoder Size': 1024,
                                    'Estimator Depth': 1,
                                    'Estimator Size': 4096,
                                    'Shared-Weights': True}

if __name__ == '__main__':
    langs = ['en', 'ro', 'ru', 'de']

    # Dictionary for storing model performance
    performance_df = pd.DataFrame(columns=['embedder_name', 'run', 'best_validation_correlation', 'peak_epoch'])
   

    # Load Dataset
    dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/dev',
                        langs=langs)
    # Make folds
    indices = torch.randperm(len(dataset)).tolist()
    val_size = 300
    data_splits = [(indices[val_size:], indices[:val_size])]

    for embedder_name, embedder, settings in embedder_iter(static_emb_path='/home/norrman/GitHub/RND/data/embeddings/multilingual_embeddings.tar.gz',
                                                 langs=langs):
        print(f'Commencing parameter search for {embedder_name}')
        best_params = hyper_parameter_search(dataset=dataset,
                                             data_splits=data_splits,
                                             n_trials=72,
                                             shared_encoder_weights=True,
                                             embedder=embedder,
                                             encoder_dim=1024,
                                             encoder_depth=settings['Encoder Depth'],
                                             estimator_hidden_size=4096,
                                             estimator_hidden_layers=settings['Estimator Depth'],
                                             max_epochs=10,
                                             verbose=False,
                                             )
        
        param_df = pd.DataFrame(best_params, index=[0])
        param_df.to_csv(f'{embedder_name}_best_params.csv')