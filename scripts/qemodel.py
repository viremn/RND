from multilingual_embeddings import MultilingualStaticSentenceEmbedder, \
    LASERSentenceEmbedder, DistilUSEEmbedder, ParaphraseEmbedder, \
    LaBSEEmbedder, BertSentenceEmbedder, XLMREmbedder

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, depth=1) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        for _ in range(1, depth):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)

        self.to(self.device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    


class Predictor(nn.Module):
    def __init__(self, embedder, encoder_size, encoder_depth=1, shared_encoder_weights=False) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = embedder
        self.embedding_size = embedder.outdim
        self.outdim = 2 * encoder_size
        
        self.src_encoder = Encoder(self.embedding_size, encoder_size, encoder_depth)
        self.tgt_encoder = self.src_encoder if shared_encoder_weights \
            else Encoder(self.embedding_size, encoder_size, encoder_depth)
        
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
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass
        

if __name__ == '__main__':
    embedder =  XLMREmbedder()
    predictor = Predictor(embedder,
                          encoder_size=1024, 
                          encoder_depth=1, 
                          shared_encoder_weights=True)
    
    test_data = ['this is a test', 'this is another test', 'what the .', 'what the .', 'what the .']
    langs = 'en'

    output = predictor(test_data, test_data, langs, langs)

    print(output.shape)

