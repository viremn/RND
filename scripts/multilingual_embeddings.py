import sys
import tarfile
import torch
import numpy as np

from laserembeddings import Laser

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


if __name__ == '__main__':
    embedder = MultilingualStaticSentenceEmbedder('/home/norrman/GitHub/RND/data/embeddings/multilingual_embeddings.tar.gz', langs=['en', 'ru', 'ro', 'de'])
    # embedder = LASERSentenceEmbedder()

    test_data = ['this is a test', 'this is another test', 'what the .', 'what the .', 'what the .']
    langs = 'en'


    embeddings = embedder(test_data, langs)
    print(embeddings)


