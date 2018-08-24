#!/usr/bin/python3

# the class for the language to get the word-bag functions
class lang:
    def __init__(self, language):
        self.language = language
        self.word2index = {'<eos>': 1, '<sos>': 2, '<pad>': 3, '<unk>': 4}
        self.index2word = {1: '<eos>', 2: '<sos>', 3: '<pad>', 4: '<unk>'}
        self.index_length = 4

    def if_exist(self, word):
        if self.word2index.get(word, None): return True
        else: return False

    def add_word(self, word):
        self.index_length += 1
        self.word2index[word] = self.index_length
        self.index2word[self.index_length] = word

    def stoi(self, word):
        # word to index
        if self.exist(word): return self.word2index[word]
        else: return '<unk>'

    def itos(self, index):
        # index to word
        if self.index2word.get(index, None): return self.index2word[index]
        else: return '<unk>'

def create_word_bag():
    # create the word-bag vector for the language
    # read the file <train-zh.txt> and <train-en.txt>, unk stands for the undefined word
    zh_lang = lang('zh')
    en_lang = lang('en')
    with open('data/train-zh.txt') as f:
        for line in f.readlines():
            for char in list(lines):
                if zh_lang.if_exist(char) or char in [' ', '\t']: continue
                else: zh_lang.add_word(char)

    with open('data/train-en.txt') as f:
        for line in f.readlines():
            for char in list(lines):
                if en_lang.if_exist(char) or char in [' ', '\t']: continue
                else: en_lang.add_word(char)

    return zh_lang, en_lang

def load_dataset(batch_size=20):
    '''
    There are 100000 sents in the train-dataset, 400 sents in the val-dataset, 1000 sents in the test-dataset
    so the batch_size should be the gcd of 400, 1000, 100000, such as [10, 20, 50, ...]
    and default to be 20
    '''
    ZH, EN = create_word_bag()
    for 

if __name__ == "__main__":
    pass
