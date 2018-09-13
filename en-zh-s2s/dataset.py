#!/usr/bin/python3


import random
import torch
import torch.nn as nn
import ipdb

class Corpus:
    '''load all the data from the corpus 
    and create the word bag model'''

    def __init__(self, path):
        # path: data
        self.test, self.val, self.train = \
            {'src': [], 'target': []}, \
            {'src': [], 'target': []}, \
            {'src': [], 'target': []}
        
        # self.load_data(path + 'test', self.test)
        self.load_data(path + 'cmn', self.train)
        # self.load_data(path + 'val', self.val)
        print('[!] load dataset over')

        # load the word bag
        # self.zh_size, self.en_size
        # self.en_stoi, self.en_itos, self.zh_stoi, self.zh_itos
        self.init_word_bag()
        print('[!] init word bag over')

        # convert the sentence into the index number
        # self.idx_test, self.idx_val, self.idx_train
        self.convert()
        print('[!] convert all the sentence')

    def convert(self, decay=True):
        # convert the sentence into the index number
        # decay decide whether delete the origin sentence
        self.idx_test, self.idx_val, self.idx_train = \
            {'src': [], 'target': []}, \
            {'src': [], 'target': []}, \
            {'src': [], 'target': []}
        
        def _helper_zh(sentence):
            unk = self.zh_stoi['<unk>']
            ready = [self.zh_stoi.get(word.strip(), unk) for word in list(sentence) if word.strip()] + [self.zh_stoi['<eos>']]
            ready[0: 0] = [self.zh_stoi['<sos>']]
            return ready

        def _helper_en(sentence):
            unk = self.en_stoi['<unk>']
            ready = [self.en_stoi.get(word.strip(), unk) for word in sentence.split() if word.strip()] + [self.en_stoi['<eos>']]
            ready[0: 0] = [self.en_stoi['<sos>']]
            return ready

        # for example in self.test['src']:
        #     self.idx_test['src'].append(_helper_en(example))
        # for example in self.test['target']:
        #     self.idx_test['target'].append(_helper_zh(example))
        # for example in self.val['src']:
        #     self.idx_val['src'].append(_helper_en(example))
        # for example in self.val['target']:
        #     self.idx_val['target'].append(_helper_zh(example))
        for example in self.train['src']:
            self.idx_train['src'].append(_helper_en(example))
        for example in self.train['target']:
            self.idx_train['target'].append(_helper_zh(example))

        if decay:
            del self.test
            del self.val
            del self.train

    def load_data(self, filename, container):
        with open(filename + '-en.txt') as f:
            for line in f:
                container['src'].append(line)

        with open(filename + '-zh.txt') as f:
            for line in f:
                container['target'].append(line)

    def init_word_bag(self):
        # special word bag, only from train dataset
        self.en_stoi, self.en_itos, self.zh_stoi, self.zh_itos = {}, [], {}, []
        self.special_word = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.zh_size, self.en_size = 4, 4    # include 4 special token
        
        # init
        for idx, item in enumerate(self.special_word):
            self.en_stoi[item] = idx
            self.en_itos.append(item)
            self.zh_stoi[item] = idx
            self.zh_itos.append(item)

        for sent in self.train['target']:
            # use jieba to cut
            # for word in jieba.cut(sent):
            for word in list(sent):
                if word.strip():
                    if word not in self.zh_stoi:
                        self.zh_size += 1
                        self.zh_itos.append(word)
                    self.zh_stoi.setdefault(word, self.zh_size - 1)
        
        for sent in self.train['src']:
            for word in sent.split():
                if word.strip():
                    if word not in self.en_stoi:
                        self.en_size += 1
                        self.en_itos.append(word)
                    self.en_stoi.setdefault(word, self.en_size - 1)

        return self.en_stoi, self.en_itos, self.zh_stoi, self.zh_itos

class Batch:
    '''
    batch data class
    '''
    def __init__(self, src, target, unk):
        self.src = src
        self.target = target
        # pad the data
        self.pad_data(unk)

    def pad_data(self, unk):
        # pad the sentence to the longest length of the batch
        max_len_src = max([len(item) for item in self.src])
        max_len_target = max([len(item) for item in self.target])

        # pad the src - English
        for item in self.src:
            item.extend([unk] * (max_len_src - len(item)))

        # pad the target - Chinese
        for item in self.target:
            item.extend([unk] * (max_len_target - len(item)))

    def get_src(self):
        # return batch tensor [seq, batch]
        # do not need to assert the CUDA
        return torch.LongTensor(self.src).transpose(0, 1)

    def get_target(self):
        # return batch tensor [seq, bacth]
        return torch.LongTensor(self.target).transpose(0, 1)

def load_dataset(batch_size=16, shuffle=True):
    # iter is the list of the batch class's instance
    corpus = Corpus('./data/')
    unk = corpus.en_stoi['<pad>']

    def _helper(data):
        begin = 0
        data_iter = []
        while begin < len(data['src']):
            data_iter.append(Batch(data['src'][begin: begin + batch_size], \
                    data['target'][begin: begin + batch_size], unk))
            begin += batch_size
        return data_iter

    train_iter, test_iter, val_iter = _helper(corpus.idx_train), _helper(corpus.idx_test), _helper(corpus.idx_val)

    if shuffle:
        random.shuffle(train_iter)
        random.shuffle(test_iter)
        random.shuffle(val_iter)

    print('[!] load dataset over, return batch data for the model')

    return train_iter, test_iter, val_iter, corpus
    
if __name__ == "__main__":
    # test process
    train_iter, test_iter, val_iter, corpus = load_dataset()
    ipdb.set_trace()
    test_batch = train_iter[1]
    src = test_batch.get_src()
    target = test_batch.get_target()
