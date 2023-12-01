from multilingual_embeddings import MultilingualStaticSentenceEmbedder, \
    LASERSentenceEmbedder, DistilUSEEmbedder, ParaphraseEmbedder, \
    LaBSEEmbedder, BertSentenceEmbedder, XLMREmbedder

from math import isnan

import optuna
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

from torch.utils.data import DataLoader, Subset
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from scipy.stats.mstats import pearsonr

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, depth=1) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layers = [nn.Linear(input_size, hidden_size), nn.LeakyReLU()]

        for _ in range(1, depth):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        self.layers = nn.ModuleList(layers)

        self.to(self.device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class Predictor(nn.Module):
    def __init__(self, 
                 encoder_dim, 
                 encoder_depth=1, 
                 shared_encoder_weights=False, 
                 embedder=None):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = XLMREmbedder() if embedder is None else embedder
        self.embedding_size = embedder.outdim
        self.outdim = 2 * encoder_dim
        
        self.src_encoder = Encoder(self.embedding_size, encoder_dim, encoder_depth)
        self.tgt_encoder = self.src_encoder if shared_encoder_weights \
            else Encoder(self.embedding_size, encoder_dim, encoder_depth)
        
        self.to(self.device)
        
    def forward(self, source_sents, target_sents, source_langs=None, target_langs=None):
        src_input = self.embedder(source_sents, source_langs)
        src_input = src_input.to(self.device)
        src_output = self.src_encoder(src_input)

        tgt_input = self.embedder(target_sents, target_langs)
        tgt_input = tgt_input.to(self.device)
        tgt_output = self.tgt_encoder(tgt_input)

        return torch.cat((src_output, tgt_output), dim=1)


class Estimator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers=2, dropout=0.2) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        layers = [nn.Linear(input_size, hidden_size), nn.LeakyReLU()]

        for i in range(hidden_layers):
            layers.append(nn.Linear(hidden_size//(2**i), hidden_size//(2**(i+1))))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_size//(2**hidden_layers), 1))
        
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)

        self.to(self.device)

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(layer(x))
        return x     


class QEModel(nn.Module):
    def __init__(self,
                 encoder_dim=1024,
                 encoder_depth=1,
                 shared_encoder_weights=True,
                 embedder=None,
                 estimator_hidden_size=2048,
                 estimator_hidden_layers=2,
                 dropout=0.2):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.predictor = Predictor(encoder_dim=encoder_dim, 
                                   encoder_depth=encoder_depth, 
                                   shared_encoder_weights=shared_encoder_weights,
                                   embedder=embedder)
        
        self.estimator = Estimator(input_size=self.predictor.outdim, 
                              hidden_size=estimator_hidden_size, 
                              hidden_layers=estimator_hidden_layers,
                              dropout=dropout)
        
        self.to(self.device)
    
    def forward(self, source_sents, target_sents, source_langs=None, target_langs=None):
        predictor_output = self.predictor(source_sents=source_sents,
                                          target_sents=target_sents, 
                                          source_langs=source_langs,
                                          target_langs=target_langs)
        estimator_output = self.estimator(predictor_output)
        return torch.sigmoid(estimator_output)*100
    

def train_model(model, 
                dataset, 
                validation_dataset=None,
                validation_size=300, 
                batch_size=128, 
                max_epochs=10, 
                learning_rate=1e-3, 
                weight_decay=1e-5,
                lr_scheduler_gamma=0.1,
                schedule_lr=True,
                collate_fn=None,
                checkpoint=None,
                outdir='',
                outname='model',
                save_latest_checkpoint=True,
                save_best_checkpoint=True,
                save_current_checkpoint=False,
                verbose=True):
    
    if isinstance(outdir, str) and not outdir.endswith('/'):
        outdir += '/'

    optimizer = optim.AdamW(model.parameters(), 
                            lr=learning_rate, 
                            weight_decay=weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                     mode='min',
                                     factor=lr_scheduler_gamma,
                                     patience=1)

    criterion = nn.MSELoss()

    history = {'train_loss': [], 'validation_loss': [], 'train_correlation': [], 'validation_correlation': [], 'lr': []}
    start_epoch = 0
    
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1

    if hasattr(dataset, 'collate_fn'):
        collate_fn = dataset.collate_fn

    if not validation_dataset and validation_size:
        train_size = len(dataset) - validation_size
        indices = torch.randperm(len(dataset)).tolist()
        train_dataset = Subset(dataset, indices[:train_size])
        validation_dataset = Subset(dataset, indices[train_size:])
    else:
        train_dataset = dataset

    bad_start = False
    complete = False
    while not complete:
        for epoch in range(start_epoch, max_epochs):
            if bad_start:
                break
            
            model.train()

            current_lr = [groups['lr'] for groups in optimizer.param_groups][0]
            history['lr'].append(current_lr)

            train_loss = 0.0
            running_output = list()
            running_labels = list()

            train_dataloader = DataLoader(train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        collate_fn=collate_fn, 
                                        drop_last=False)
            batches = tqdm(train_dataloader, 
                            desc=f'Epoch {epoch+1}',
                            unit='batch', 
                            total=(len(train_dataset)//batch_size)+(1 if len(train_dataset)%batch_size else 0),
                            disable=not verbose)

            for i, batch in enumerate(batches):
                output = model(batch['original'].tolist(),
                                batch['translation'].tolist(),
                                batch['original_lang'].tolist(),
                                batch['translation_lang'].tolist())
                
                labels = torch.tensor(batch['mean'].tolist(), dtype=torch.float).to(model.device)

                optimizer.zero_grad()

                loss = criterion(output.squeeze(), labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(output)
                running_labels.extend(labels.reshape(-1,).detach().cpu().numpy())
                running_output.extend(output.reshape(-1,).detach().cpu().numpy())
                running_correlation, _ = pearsonr(running_output, running_labels)
            

                batches.set_postfix({'Avg_loss': train_loss/(i*batch_size+len(output)),
                                    'correlation': running_correlation,
                                    'lr': current_lr})
            
            history['train_loss'].append(train_loss)
            history['train_correlation'].append(running_correlation)

            if validation_dataset:
                model.eval()
                with torch.no_grad():
                    validation_dataloader = DataLoader(validation_dataset, 
                                                    batch_size=batch_size,
                                                    collate_fn=collate_fn, 
                                                    drop_last=False)
                    
                    val_loss = 0.0
                    running_val_output = list()
                    running_val_labels = list()

                    batches = tqdm(validation_dataloader, 
                                    desc=f'Validation {epoch+1}', 
                                    unit='batch', 
                                    total=(len(validation_dataset)//batch_size)+(1 if len(validation_dataset)%batch_size else 0),
                                    disable=not verbose)

                    for i, batch in enumerate(batches):
                        output = model(batch['original'].tolist(),
                                    batch['translation'].tolist(),
                                    batch['original_lang'].tolist(),
                                    batch['translation_lang'].tolist())
                        
                        labels = torch.tensor(batch['mean'].tolist(), dtype=torch.float).to(model.device)
                        
                        
                        loss = criterion(output.squeeze(), labels)

                        val_loss += loss.item() * len(output)
                        running_val_labels.extend(labels.reshape(-1,).detach().cpu().numpy())
                        running_val_output.extend(output.reshape(-1,).detach().cpu().numpy())
                        running_val_correlation, _ = pearsonr(running_val_output, running_val_labels)

                        if isnan(running_val_correlation):
                            bad_start = True
                            break

                        if not isinstance(running_val_correlation, float):
                            print(running_val_output)
                            print(running_val_labels)

                        batches.set_postfix({'Avg_loss': val_loss/(i*batch_size+len(output)),
                                                'correlation': running_val_correlation})

                    history['validation_loss'].append(val_loss)
                    history['validation_correlation'].append(running_val_correlation)

                    

            if schedule_lr:
                lr_scheduler.step(history['validation_loss'][-1])

            if save_latest_checkpoint:
                model_path = outdir + outname + f'.last.checkpoint.pt'
                checkpoint = {'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'history': history}
                torch.save(checkpoint, model_path)
            
            if save_best_checkpoint:
                if validation_dataset and \
                    max(history['validation_correlation']) == history['validation_correlation'][-1]:
                        model_path = outdir + outname + f'.best.checkpoint.pt'
                        checkpoint = {'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'history': history}
                        torch.save(checkpoint, model_path)
            
            if save_current_checkpoint:
                model_path = outdir + outname + f'.epoch-{epoch}.checkpoint.pt'
                checkpoint = {'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'history': history}
                torch.save(checkpoint, model_path)
        if not bad_start:
            complete = True
        else:
            print('ERROR: BAD START. Restarting training...', file=sys.stderr)
            bad_start = False
            model.apply(lambda m: m.reset_parameters() if isinstance(m, nn.Linear) else m)
            optimizer = optim.AdamW(model.parameters(), 
                            lr=learning_rate, 
                            weight_decay=weight_decay)
            lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                            mode='min',
                                            factor=lr_scheduler_gamma,
                                            patience=1)
            history = {'train_loss': [], 'validation_loss': [], 'train_correlation': [], 'validation_correlation': [], 'lr': []}

    return history

def eval_model(model,
               dataset,
               batch_size=128,
               collate_fn=None):
    
    if hasattr(dataset, 'collate_fn'):
        collate_fn = dataset.collate_fn
    
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        validation_dataloader = DataLoader(validation_dataset, 
                                           batch_size=batch_size,
                                           collate_fn=collate_fn, 
                                           drop_last=False)
        
        val_loss = 0.0
        running_val_output = list()
        running_val_labels = list()

        batches = tqdm(validation_dataloader, 
                       desc=f'Evaluation', 
                       unit='batch', 
                       total=(len(validation_dataset)//batch_size)+(1 if len(validation_dataset)%batch_size else 0))

        for i, batch in enumerate(batches):
            output = model(batch['original'].tolist(),
                           batch['translation'].tolist(),
                           batch['original_lang'].tolist(),
                           batch['translation_lang'].tolist())
            
            labels = torch.tensor(batch['mean'].tolist(), dtype=torch.float).to(model.device)           

            loss = criterion(output.squeeze(), labels)

            val_loss += loss.item() * len(output)
            running_val_labels.extend(labels.reshape(-1,).detach().cpu().numpy())
            running_val_output.extend(output.reshape(-1,).detach().cpu().numpy())
            running_val_correlation, running_p = pearsonr(running_val_output, running_val_labels)

            batches.set_postfix({'Avg_loss': val_loss/(i*batch_size+len(output)),
                                 'correlation': running_val_correlation,
                                 'p': running_p})

    print(f'Evaluation complete!\nTotal Loss: {val_loss}\nAverage Loss: {val_loss/len(dataset)}\nPearson r correlation: {running_val_correlation}\nP-value: {running_p}')
    return {'total_loss': val_loss, 'avg_loss': val_loss/len(dataset), 'correlation': running_val_correlation, 'p': running_p}

def predict(model,
            original_sents,
            translation_sents,
            original_langs=None,
            translation_langs=None):
    model.eval()                        
    predictions = model(original_sents,
                        translation_sents,
                        original_langs,
                        translation_langs)
    return pd.DataFrame({'original': original_sents, 
               'translation': translation_sents, 
               'predictions': predictions.tolist()})
    
def objective(trial, dataset, data_splits, settings):
    # Existing hyperparameters
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    lr_scheduler_gamma = trial.suggest_float('lr_scheduler_gamma', 0.1, 0.9, step=0.1)

    print(f'{dropout=:.1g} {learning_rate=:.1g} {weight_decay=:.1g} {lr_scheduler_gamma=:.1g}')

    performance = list()

    folds = tqdm(data_splits, desc=f'Trial {trial.number}', unit='fold', total=len(data_splits))

    for train_indices, validation_indices in folds:
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
        train_dataset.collate_fn = dataset.collate_fn
        validation_dataset.collate_fn = dataset.collate_fn

        model = QEModel(encoder_dim=settings['encoder_dim'], 
                        encoder_depth=settings['encoder_depth'], 
                        shared_encoder_weights=settings['shared_encoder_weights'],
                        embedder=settings['embedder'],
                        estimator_hidden_size=settings['estimator_hidden_size'], 
                        estimator_hidden_layers=settings['estimator_hidden_layers'],
                        dropout=dropout)
        
        history = train_model(model=model,
                            dataset=train_dataset,
                            validation_dataset=validation_dataset,
                            validation_size=None, 
                            batch_size=settings['batch_size'], 
                            max_epochs=settings['max_epochs'],
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            lr_scheduler_gamma=lr_scheduler_gamma,
                            schedule_lr=True,
                            save_best_checkpoint=False,
                            save_current_checkpoint=False,
                            save_latest_checkpoint=False,
                            checkpoint=None,
                            verbose=settings['verbose'])

        performance.append(max(history['validation_correlation']))

        folds.set_postfix({'avg_corr': sum(performance)/len(performance)})
    
    return sum(performance)/len(performance)

def hyper_parameter_search(dataset, 
                           data_splits, 
                           n_trials, 
                           encoder_dim=1024,
                           encoder_depth=1,
                           shared_encoder_weights=True,
                           embedder=XLMREmbedder(),
                           estimator_hidden_size=4096,
                           estimator_hidden_layers=2,
                           batch_size=32, 
                           max_epochs=20,
                           verbose=True):
    
    settings = {'encoder_dim': encoder_dim,
                'encoder_depth': encoder_depth,
                'shared_encoder_weights': shared_encoder_weights,
                'embedder': embedder,
                'estimator_hidden_size': estimator_hidden_size,
                'estimator_hidden_layers': estimator_hidden_layers,
                'batch_size': batch_size,
                'max_epochs': max_epochs,
                'verbose': verbose}

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dataset, data_splits, settings), n_trials=n_trials)

    best_params = study.best_params
    best_accuracy = study.best_value

    print(f"Best hyperparameters: {best_params} with accuracy: {best_accuracy}")
    
    return best_params
    
def cross_fold_splits(dataset, num_splits):
    indices = torch.randperm(len(dataset)).tolist()
    validation_size = len(dataset) // num_splits
    data_splits = list()
    for i in range(0, len(dataset), validation_size):
        validation_indices = indices[i:i+validation_size]
        train_indices = indices[:i] + indices[i+validation_size:] 
        data_splits.append((train_indices, validation_indices))

        assert len(validation_indices) + len(train_indices) == len(dataset)
    
    return data_splits

if __name__ == '__main__':

    from load_data import QEDataset
    from torch.utils.data import DataLoader


    embedding_path = "/home/norrman/GitHub/RND/data/embeddings/multilingual_embeddings.tar.gz"
    data_path = "/home/norrman/GitHub/RND/data/direct-assessments/"
    langs = "en", "de", "ro" , "ru"

    outdir = "/home/norrman/GitHub/RND/models/"
    outname = "test_model"


    print('Loading Dataset...')
    train_dataset = QEDataset(data_path+'dev', langs)
    validation_dataset = QEDataset(data_path+'test', langs)

    print('Loading Model...')
    # embedder =  MultilingualStaticSentenceEmbedder(embedding_file_path=embedding_path, langs=langs)
    embedder = XLMREmbedder()


    # data_splits = cross_fold_splits(train_dataset, num_splits=5)
    # hyper_parameter_search(train_dataset, 
    #                        data_splits=data_splits,
    #                        n_trials=5,
    #                        max_epochs=5,
    #                        verbose=False)

    model = QEModel(encoder_dim=1024, 
                    encoder_depth=2, 
                    shared_encoder_weights=True,
                    embedder=embedder,
                    estimator_hidden_size=4096, 
                    estimator_hidden_layers=1,
                    dropout=0.2)
    
    checkpoint = None
    # checkpoint = torch.load('/home/norrman/GitHub/RND/models/test_model.best.checkpoint.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])

    print('Training Model...')
    train_model(model=model,
                dataset=train_dataset,
                validation_dataset=None,
                validation_size=300, 
                batch_size=16, 
                max_epochs=10,
                outdir=outdir,
                outname=outname,
                save_best_checkpoint=True,
                save_current_checkpoint=False,
                save_latest_checkpoint=True,
                checkpoint=checkpoint,
                verbose=True)
    
    eval_model(model=model,
               dataset=validation_dataset)
    
    # print(predict(model=model,
    #               original_sents=validation_dataset.data['original'].tolist(),
    #               translation_sents=validation_dataset.data['translation'].tolist()))
    
