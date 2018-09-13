#!/usr/bin/python3

'''handle the val and test dataset into the format of the train dataset'''

import jieba

def handle(filename):
    with open(filename + '.txt') as f:
        zh, en = [], []
        for line in f:
            x, y = line.split('\t')
            zh.append(y)
            en.append(x)
        
    # write into zh
    with open(filename + '-zh.txt', 'w') as f:
        for line in zh:
            f.write(line[:-2] + '\n')
    # write into en
    with open(filename + '-en.txt', 'w') as f:
        for line in en:
            f.write(line[:-1] + '\n')
    
    print(f'write the data into file [{filename}-zh.txt], [{filename}-en.txt]')

def test_max_line(filename):
    '''test the max length of the line in the CHINESE LANGUAGE file'''
    with open(filename + '-zh.txt') as f:
        max_length, cache, max_idx = -1, '', 0
        for idx, sent in enumerate(f.readlines()):
            words = list(jieba.cut(''.join(sent.split())))
            if max_length < len(words):
                max_length = len(words)
                max_idx= idx
                cache = sent
    print(filename + '-zh.txt max length:', max_length)
    print(max_idx, '\n', cache)
    print(list(jieba.cut(''.join(cache.split()))))

if __name__ == "__main__":
    handle('cmn')
    # test_max_line('val')
