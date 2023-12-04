from qemodel import QEModel, train_model, eval_model, predict, cross_fold_splits, hyper_parameter_search

from multilingual_embeddings import MultilingualStaticSentenceEmbedder, \
    LASERSentenceEmbedder, DistilUSEEmbedder, ParaphraseEmbedder, \
    LaBSEEmbedder, BertSentenceEmbedder, XLMREmbedder

from load_data import QEDataset

from torch.utils.data import Subset



import pandas as pd
import torch
import json

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
    settings = {'Learning Rate': 1.00E-03,
                'Weight Decay': 1.00E-05,
                'Dropout': 0.2,
                'LR Scheduler Factor': 0.1,
                'Encoder Depth': 2,
                'Encoder Size': 1024,
                'Estimator Depth': 1,
                'Estimator Size': 4096,
                'Shared-Weights': True}
    
    langs = ['en', 'ro', 'ru', 'de']

    outdir = "/home/norrman/GitHub/RND/models/"

    # Dictionary for storing model performance
    performance_df = pd.DataFrame(columns=['embedder_name', 'run', 'best_validation_correlation', 'peak_epoch'])


    # Load Dataset
    dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/train',
                        langs=langs)
    

    # Make folds
    indices = torch.randperm(len(dataset)).tolist()
    val_size = 1000
    train_indices, validation_indices = indices[val_size:], indices[:val_size]

    train_dataset = Subset(dataset, indices=train_indices)
    train_dataset.collate_fn = dataset.collate_fn
    val_dataset = Subset(dataset, indices=validation_indices)
    val_dataset.collate_fn = dataset.collate_fn

    with open('validation_dataset_info.json', 'w') as f:
        validation_data = [dataset[i] for i in validation_indices]
        validation_data = {i: {'original': item['original'],
                            'translastion': item['translation'],
                            'score': float(item['mean']),
                            'original_lang': item['original_lang'],
                            'translation_lang': item['translation_lang']} for i, item in enumerate(validation_data)}
        data_distribution = dict()
        for value in validation_data.values():
            key = f'{value["original_lang"]}-{value["translation_lang"]}'
            if key not in data_distribution.keys():
                data_distribution[key] = 0
            data_distribution[key] += 1

        validation_data.update(data_distribution)

        json.dump(validation_data, f, indent=4)

    embedders = embedder_iter(static_emb_path='/home/norrman/GitHub/RND/data/embeddings/multilingual_embeddings.tar.gz',
                               langs=langs)
    for embedder_name, embedder in embedders:

        outname = f'{embedder_name}_deep_encoder_shallow_estimator_uniform'

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
                            max_epochs=1000,
                            outdir=outdir,
                            outname=outname,
                            learning_rate=settings['Learning Rate'],
                            weight_decay=settings['Weight Decay'],
                            save_latest_checkpoint=True,
                            save_best_checkpoint=True,
                            save_current_checkpoint=False,
                            verbose=True,
                            terminate_on_lr=1e-7)
        
