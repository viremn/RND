import sys
import tarfile
import torch
import numpy as np

from laserembeddings import Laser
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertModel

class MultilingualEmbeddings:
    def __init__(self, path, lang) -> None:
        self.embeddings = dict()
        self.get_embeddings(path, lang)

    def get_embeddings(self, file, lang):
        print(f'Loading {lang} embeddings from file', file=sys.stderr)
        with tarfile.open(file, 'r') as f:
            content = f.extractfile([member for member in f.getmembers() if member.name.endswith(lang)][0])
            for line in content:
                word, embedding = line.decode().split(maxsplit=1)
                embedding = np.array(embedding.strip().split(), dtype=np.float32)
                self.embeddings[word] = embedding

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return self.embeddings.get(index, torch.zeros([300]))
    
    def embed(self, tokens, return_tensor=False):
        if isinstance(tokens, str):
            tokens = tokens.split(' ')
            tokens = [np.vstack([self[t] for t in tokens])]
        elif isinstance(tokens, list):
            if all(isinstance(item, str) for item in tokens):
                if any(' ' in item for item in tokens):
                    tokens = [item.split(' ') for item in tokens]
                    tokens = [np.vstack([self[t] for t in seq]) for seq in tokens]
            elif all(isinstance(item, list) for item in tokens):
                tokens = [np.vstack([self[t] for t in seq]) for seq in tokens]
            else:
                tokens = [np.vstack([self[t] for t in tokens])]
    
        if return_tensor:
            tokens = [torch.from_numpy(seq) for seq in tokens]
        if len(tokens) > 1:
            return tokens
        else:
            return tokens[0]
        
class MultilingualStaticSentenceEmbedder:
    def __init__(self, embedding_file_path, langs) -> None:
        self.__setup(embedding_file_path, langs)
        self.outdim = 300

    def __setup(self, embedding_file_path, langs):
        self.embeddings = dict()
        for lang in langs:
            self.embeddings[lang] = MultilingualEmbeddings(embedding_file_path, lang)
        self.langs = langs
    
    def __len__(self):
        return sum(len(self.embeddings[key]) for key in self.embeddings)

    def __call__(self, sents, langs):
        if isinstance(langs, str):
            langs = [langs for _ in range(len(sents))]
        embedded_sents = list()
        for sent, lang in zip(sents, langs):
            embedding = torch.mean(self.embeddings[lang].embed(sent, return_tensor=True), dim=0)
            embedded_sents.append(embedding)
        return torch.vstack(embedded_sents)

class LASERSentenceEmbedder:
    def __init__(self) -> None:
        self.embedder = Laser()
        self.outdim = 1024

    def __call__(self, sents, langs):
        return torch.from_numpy(self.embedder.embed_sentences(sents, lang=langs))

class DistilUSEEmbedder:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2', device=self.device)
        for p in self.embedder.parameters():
            p.requires_grad = False
        self.outdim = 512

    def __call__(self, sents, langs=None):
        return self.embedder.encode(sents, convert_to_tensor=True)
    
class XLMREmbedder:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SentenceTransformer('xlm-roberta-large', device=self.device)
        for p in self.embedder.parameters():
            p.requires_grad = False
        self.outdim = 768

    def __call__(self, sents, langs=None):
        return self.embedder.encode(sents, convert_to_tensor=True)

class ParaphraseEmbedder:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=self.device)
        for p in self.embedder.parameters():
            p.requires_grad = False
        self.outdim = 768

    def __call__(self, sents, langs=None):
        return self.embedder.encode(sents, convert_to_tensor=True)

class LaBSEEmbedder:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SentenceTransformer('LaBSE', device=self.device)
        for p in self.embedder.parameters():
            p.requires_grad = False
        self.outdim = 768

    def __call__(self, sents, langs=None):
        return self.embedder.encode(sents, convert_to_tensor=True)

class BertSentenceEmbedder:
    def __init__(self, pooling='cls') -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        self.embedder = BertModel.from_pretrained('bert-base-multilingual-cased')
        for p in self.embedder.parameters():
            p.requires_grad = False
        self.pooling = pooling
        self.outdim = 768

        self.embedder.to(self.device)

    def __call__(self, sents, langs=None):
        tokenized_sents = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        tokenized_sents.to(self.device)
        output = self.embedder(**tokenized_sents)

        if self.pooling == 'cls':
            output = output.last_hidden_state[:, 0, :]
        elif self.pooling == 'max':
            output = torch.max(output.last_hidden_state, dim=1)[0]
        elif self.pooling == 'mean':
            output = torch.mean(output.last_hidden_state, dim=1)

        return output


if __name__ == '__main__':
    # embedder = MultilingualStaticSentenceEmbedder('/home/norrman/GitHub/RND/data/embeddings/multilingual_embeddings.tar.gz', langs=['en', 'ru', 'ro', 'de'])
    # embedder = LASERSentenceEmbedder()
    embedder = XLMREmbedder()

    test_data = ['this is a test', 'this is another test', 'what the .', 'what the .', 'what the .']
    langs = 'en'


    embeddings = embedder(test_data, langs)
    print(embeddings)
    print(embeddings.shape)


