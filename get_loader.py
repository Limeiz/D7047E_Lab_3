import os
import random
from itertools import chain

import pandas as pd
import spacy 
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
import torchvision.transforms.v2 as transforms


# convert text -> numerical values
# Setup padding of every batch (all examples should be of same seq_len and setup dataloader)

spacy_eng = spacy.load('en_core_web_sm')  # _md

# from sentence_transformers import SentenceTransformer, util



class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.vector = None
        self.freq_threshold = freq_threshold
        self.weights = 1
#        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return spacy_eng.tokenizer(text)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        vector = []
        for sentence in sentence_list:
            for tok in self.tokenizer_eng(sentence):
                word = tok.text.lower()
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
#                    vector.append(tok.vector)
                    idx += 1
#        self.vector = torch.tensor(vector)
        self.weights = torch.zeros(len(self.stoi))
        for idx, word in self.itos.items():
            frequency = frequencies.get(word, self.freq_threshold)  # special symbols -> threshold
            self.weights[idx] = frequency
        max_frequency = torch.max(self.weights)
        self.weights = 1 + torch.log(max_frequency) - torch.log(self.weights)

#    def vectorize(self, sentence):
#        v_sum = torch.zeros_like(self.vector[0])
#        for idx in sentence:
#            if idx < 4:
#                continue  # special
#            v = self.vector[idx - 4]
#            v_sum += v
#        return v_sum / len(sentence)

    def numericalize(self, text):
        tokens = self.tokenizer_eng(text)
        words = [token.text.lower() for token in tokens]

        return [
            self.stoi[word] if word in self.stoi else self.stoi["<UNK>"]
            for word in words
        ]

    def stringify(self, tokenized_text):
        return " ".join([self.itos[token] for token in tokenized_text])


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transformers=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

        # Prepare transformers
        self.transformers = {} if transformers is None else transformers

    def get_loader(self, split, transform=None, *args, **kwargs):
        indexes = None
        return DataSplit(self, split=split, indexes=indexes, transform=transform, *args, **kwargs)


class CocoDataset(Dataset):
    def __init__(self, root_dir, captions_file,
                 transform=None, transformers=None, freq_threshold=5,
                 *args, **kwargs):
        """

        :param root_dir:
        :param captions_file:
        :param transform: plain image to tensor transformer, no agumentation
        :param transformers: dictionary of transformers, one for each split
        :param freq_threshold:
        """
        self.root_dir = root_dir
        df = pd.read_json(captions_file)

        # Get img, caption columns
        self.imgs = []
        self.captions = []
        self.split = dict()
        self.img_next = 0
        for desc in df["images"]:
            dir = desc.get('filepath')
            full_file_name = dir + os.sep + desc.get('filename')
            imgid = desc.get('imgid')
            assert imgid == self.img_next
            self.img_next += 1
            alt_sentences = list(map(lambda s: s['raw'], desc.get("sentences")))
            self.imgs.append(full_file_name)
            self.captions.append(alt_sentences)

            split = desc.get('split')
            assert split in ["train", "val", "test", "restval"]
            split_ = self.split.get(split, [])
            self.split[split] = split_
            self.split[split].append(imgid)

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(chain.from_iterable(self.captions))

        # Prepare transformers
        self.transformers = transformers if transformers is not None else {
            'train': transforms.Compose([transforms.AutoAugment(),
                                         transform,
                                         ]),
            'val': transform,
            'test': transform,
        }


    def get_loader(self, split=None, transform=None, **kwargs):
        indexes = None if split is None else self.split.get(split, [])
        transform = self.transformers[split] if transform is None else transform
        return DataSplit(self, split, indexes=indexes, transform=transform)


class DataSplit(Dataset):
    def __init__(self, dataset, split, indexes=None, transform=None, *args, **kwargs):
        self.dataset = dataset
        self.split = split
        self.transform = transform
        self.indexes = indexes
        self.use_only_first_caption = False

#        for alt_sentences in self.dataset.captions:
#            sentence_info = self.dataset.vocab.sentence_transformer.encode(alt_sentences,
#                                                                   convert_to_tensor=True)
#            print([1 - util.pytorch_cos_sim(sentence_info[0], sentenceA).cpu() for sentenceA in sentence_info])

    def __len__(self):
        return len(self.indexes) if self.indexes is not None else len(self.dataset.imgs)

    def __getitem__(self, index):
        # convert to global index
        if self.indexes is not None:
            index = self.indexes[index]

        captions_ = self.dataset.captions[index]
        caption = captions_[0] if self.use_only_first_caption else random.choice(captions_)  # Randomly select one caption #
        img_id = self.dataset.imgs[index]
        img = Image.open(os.path.join(self.dataset.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.dataset.vocab.stoi["<SOS>"]]
        numericalized_caption += self.dataset.vocab.numericalize(caption)
        numericalized_caption.append(self.dataset.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_dataset(root_folder, annotation_file, *args, **kwargs):
    if "flickr" in annotation_file:
        dataset_ = FlickrDataset(root_folder, annotation_file, *args, **kwargs)
    elif "coco" in annotation_file:
        dataset_ = CocoDataset(root_folder, annotation_file, *args, **kwargs)
    else:
        raise NotImplementedError("Unexpected dataset: " + annotation_file)
    return dataset_

def get_batch_loader(
        dataset,
        split,
        transform=None,
        auto_augment=False,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    pad_idx = dataset.vocab.stoi["<PAD>"]

    if auto_augment and transform is not None:
        transform = transforms.Compose([transforms.AutoAugment(),  # RandAugment
                                        transform,
                                        ])
    loader = DataLoader(
        dataset=dataset.get_loader(split=split, transform=transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader


def main():
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(), ]
    )
    dataset = get_dataset(
        "data/flickr8k/images/",
        "data/flickr8k/captions.txt",
    )
    loader = get_batch_loader(dataset, split="val", transform=transform)

    for idx, (imgs, captions) in enumerate(loader):
        print(f"{idx:3d} / {len(loader):5}")
        print(imgs.shape)
        print(captions.shape)


if __name__ == "__main__":
    main()

