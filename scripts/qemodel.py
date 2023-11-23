from multilingual_embeddings import MultilingualStaticSentenceEmbedder, \
    LASERSentenceEmbedder, DistilUSEEmbedder, ParaphraseEmbedder, \
    LaBSEEmbedder, BertSentenceEmbedder, XLMREmbedder

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from scipy.stats import pearsonr

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
    
    def train_model(self, 
                    train_dataset, 
                    validation_dataset=None, 
                    batch_size=128, 
                    max_epochs=10, 
                    learning_rate=1e-3, 
                    weight_decay=1e-5):
        
        optimizer = optim.AdamW(self.parameters(), 
                                lr=learning_rate, 
                                weight_decay=weight_decay)
        criterion = nn.MSELoss()

        for epoch in range(max_epochs):
            self.train()

            
            train_loss = 0.0
            running_output = list()
            running_labels = list()

            train_dataloader = DataLoader(train_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          collate_fn=train_dataset.collate_fn, 
                                          drop_last=False)
            batches = tqdm(train_dataloader, 
                           desc=f'Epoch {epoch+1}',
                           unit='batch', 
                           total=(len(train_dataset)//batch_size)+(1 if len(train_dataset)%batch_size else 0))

            for i, batch in enumerate(batches):
                output = self(batch['original'].tolist(),
                              batch['translation'].tolist(),
                              batch['original_lang'].tolist(),
                              batch['translation_lang'].tolist())
                
                labels = torch.tensor(batch['mean'].tolist(), dtype=torch.float).to(self.device)

                optimizer.zero_grad()

                loss = criterion(output.squeeze(), labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(output)
                running_labels.extend(labels.reshape(-1,).detach().cpu().numpy())
                running_output.extend(output.reshape(-1,).detach().cpu().numpy())
                running_correlation, _ = pearsonr(running_output, running_labels)
            

                batches.set_postfix({'Avg_loss': train_loss/(i*batch_size+len(output)),
                                     'correlation': running_correlation})
            
            if validation_dataset:
                self.eval()
                with torch.no_grad():
                    validation_dataloader = DataLoader(validation_dataset, 
                                                       batch_size=batch_size,
                                                       collate_fn=validation_dataset.collate_fn, 
                                                       drop_last=False)
                    
                    val_loss = 0.0
                    running_val_output = list()
                    running_val_labels = list()

                    batches = tqdm(validation_dataloader, 
                                   desc=f'Validation {epoch+1}', 
                                   unit='batch', 
                                   total=(len(validation_dataset)//batch_size)+(1 if len(validation_dataset)%batch_size else 0))

                    for i, batch in enumerate(batches):
                        output = self(batch['original'].tolist(),
                                      batch['translation'].tolist(),
                                      batch['original_lang'].tolist(),
                                      batch['translation_lang'].tolist())
                        
                        labels = torch.tensor(batch['mean'].tolist(), dtype=torch.float).to(self.device)

                        loss = criterion(output.squeeze(), labels)

                        val_loss += loss.item() * len(output)
                        running_val_labels.extend(labels.reshape(-1,).detach().cpu().numpy())
                        running_val_output.extend(output.reshape(-1,).detach().cpu().numpy())
                        running_val_correlation, _ = pearsonr(running_val_output, running_val_labels)
                    

                        batches.set_postfix({'Avg_loss': val_loss/(i*batch_size+len(output)),
                                             'correlation': running_val_correlation})
                             









if __name__ == '__main__':

    from load_data import QEDataset
    from torch.utils.data import DataLoader


    embedding_path = "/home/norrman/GitHub/RND/data/embeddings/multilingual_embeddings.tar.gz"
    data_path = "/home/norrman/GitHub/RND/data/direct-assessments/"
    langs = "en", "de", "ro" , "ru"

    print('Loading Dataset...')
    train_dataset = QEDataset(data_path+'train', langs)
    validation_dataset = QEDataset(data_path+'dev', langs)

    print('Loading Model...')
    # embedder =  MultilingualStaticSentenceEmbedder(embedding_file_path=embedding_path, langs=langs)
    embedder = XLMREmbedder()

    model = QEModel(encoder_dim=1024, 
                    encoder_depth=1, 
                    shared_encoder_weights=True,
                    embedder=embedder,
                    estimator_hidden_size=4096, 
                    estimator_hidden_layers=2,
                    dropout=0.2)

    print('Training Model...')
    model.train_model(train_dataset=train_dataset,
                      validation_dataset=validation_dataset, 
                      batch_size=16, 
                      max_epochs=10)
    
