#!/usr/bin/python3

'''handle the val and test dataset into the format of the train dataset'''

def handle(filename):
    with open(filename + '.txt') as f:
        idx = 0
        zh, en = [], []
        for line in f:
            if line.strip():
                if idx % 2 == 0:
                    # zh sentence
                    zh.append(line)
                else:
                    en.append(line)
                idx += 1
        
    # write into zh
    with open(filename + '-zh.txt', 'w') as f:
        for line in zh:
            f.write(line)
    # write into en
    with open(filename + '-en.txt', 'w') as f:
        for line in en:
            f.write(line)
    
    print(f'write the data into file [{filename}-zh.txt], [{filename}-en.txt]')

if __name__ == "__main__":
    handle('test')