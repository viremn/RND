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
                                             langs=langs)
    yield 'laser', LASERSentenceEmbedder()
    yield 'distil', DistilUSEEmbedder()
    yield 'paraphrase', ParaphraseEmbedder()
    yield 'labse', LaBSEEmbedder()
    yield 'bert-cls', BertSentenceEmbedder(pooling='cls')
    yield 'bert-mean', BertSentenceEmbedder(pooling='mean')
    yield 'xlmr', XLMREmbedder()

if __name__ == '__main__':
    settings_dict = {'base': 
                        {'Learning Rate': 1.00E-03,
                         'Weight Decay': 1.00E-05,
                         'Dropout': 0.2,
                         'LR Scheduler Factor': 0.1,
                         'Encoder Depth': 1,
                         'Encoder Size': 1024,
                         'Estimator Depth': 2,
                         'Estimator Size': 4096,
                         'Shared-Weights': True},
                'deep_encoder': 
                        {'Learning Rate': 1.00E-03,
                         'Weight Decay': 1.00E-05,
                         'Dropout': 0.2,
                         'LR Scheduler Factor': 0.1,
                         'Encoder Depth': 2,
                         'Encoder Size': 1024,
                         'Estimator Depth': 2,
                         'Estimator Size': 4096,
                         'Shared-Weights': True},
                'shallow_estimator': 
                        {'Learning Rate': 1.00E-03,
                         'Weight Decay': 1.00E-05,
                         'Dropout': 0.2,
                         'LR Scheduler Factor': 0.1,
                         'Encoder Depth': 1,
                         'Encoder Size': 1024,
                         'Estimator Depth': 1,
                         'Estimator Size': 4096,
                         'Shared-Weights': True},
                'deep_encoder_shallow_estimator': 
                        {'Learning Rate': 1.00E-03,
                         'Weight Decay': 1.00E-05,
                         'Dropout': 0.2,
                         'LR Scheduler Factor': 0.1,
                         'Encoder Depth': 2,
                         'Encoder Size': 1024,
                         'Estimator Depth': 1,
                         'Estimator Size': 4096,
                         'Shared-Weights': True},
                'split_weigths': {
                         'Learning Rate': 1.00E-03,
                         'Weight Decay': 1.00E-05,
                         'Dropout': 0.2,
                         'LR Scheduler Factor': 0.1,
                         'Encoder Depth': 1,
                         'Encoder Size': 1024,
                         'Estimator Depth': 2,
                         'Estimator Size': 4096,
                         'Shared-Weights': False},
                'narrow_estimator': 
                        {'Learning Rate': 1.00E-03,
                         'Weight Decay': 1.00E-05,
                         'Dropout': 0.2,
                         'LR Scheduler Factor': 0.1,
                         'Encoder Depth': 1,
                         'Encoder Size': 1024,
                         'Estimator Depth': 2,
                         'Estimator Size': 2048,
                         'Shared-Weights': True}}
    
    langs = ['en', 'ro', 'ru', 'de']

    # Dictionary for storing model performance
    performance_df = pd.DataFrame(columns=['embedder_name', 'run', 'best_validation_correlation', 'peak_epoch'])
   

    # Load Dataset
    dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/dev',
                        langs=langs)
    # Make folds
    indices = torch.randperm(len(dataset)).tolist()
    val_size = 300
    data_splits = [(indices[:i*val_size] + indices[i*val_size+val_size:], indices[i*val_size:i*val_size+val_size]) for i in range(3)]

    for study_name, settings in settings_dict.items():
        print(f'Commencing {study_name} study...')
        for run, (train_indices, val_indices) in enumerate(data_splits):
            print(f'Commencing run {run}...')
            # Prepare Subsets
            train_dataset = Subset(dataset, indices=train_indices)
            train_dataset.collate_fn = dataset.collate_fn
            val_dataset = Subset(dataset, indices=val_indices)
            val_dataset.collate_fn = dataset.collate_fn
            
            # Initialize Embedder Iter
            embedding_types = embedder_iter(static_emb_path='/home/norrman/GitHub/RND/data/embeddings/multilingual_embeddings.tar.gz',
                                            langs=langs)
            for embedder_name, embedder in embedding_types:
                print(f'Training {embedder_name}...')
                
                torch.manual_seed(0)
                model = QEModel(embedder=embedder,
                                encoder_dim=settings['Encoder Size'],
                                encoder_depth=settings['Encoder Depth'],
                                shared_encoder_weights=settings['Shared-Weights'],
                                estimator_hidden_size=settings['Estimator Size'],
                                estimator_hidden_layers=settings['Estimator Depth'],
                                dropout=settings['Dropout'])
                history = train_model(model=model,
                                    dataset=train_dataset,
                                    validation_dataset=val_dataset,
                                    batch_size=32,
                                    max_epochs=15,
                                    learning_rate=settings['Learning Rate'],
                                    weight_decay=settings['Weight Decay'],
                                    save_latest_checkpoint=False,
                                    save_best_checkpoint=False,
                                    save_current_checkpoint=False,
                                    verbose=True)
                
                performance_df.loc[f'{embedder_name}-{run}'] = [embedder_name, 
                                                    run, 
                                                    max(history['validation_correlation']), 
                                                    history['validation_correlation'].index(max(history['validation_correlation']))]
        performance_df.to_csv(f'{study_name}_output.csv')
        performance_df = pd.DataFrame(columns=['embedder_name', 'run', 'best_validation_correlation', 'peak_epoch'])