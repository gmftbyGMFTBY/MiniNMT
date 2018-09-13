#!/usr/bin/python3

'''
Use the model to test some transformations
or test the model's score on some important metrics
'''

from dataset import load_dataset
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Encoder, Decoder, Seq2Seq
import os
import ipdb
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=16,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='clip the grad to invode the gradient explore')
    return p.parse_args()

def metrics():
    # test the model's metrics, such as BLUE score
    pass

def cli(sentence, corpus):
    # return blank sentence inpt
    if not sentence:
        print('-' * 50)
        return

    # command line to transform the input English sentence
    hidden_size, embed_size = 256, 128
    assert torch.cuda.is_available()

    zh_size, en_size = corpus.zh_size, corpus.en_size

    # load dataset
    encoder = Encoder(en_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, zh_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    seq2seq.load_state_dict(torch.load('.save/seq2seq_200.pt'))

    # model for test
    seq2seq.eval()

    # convert the sentence into the batch input to get the output
    # TODO: consider the <unk> token into the inpt
    # add the <eos> and <sos> into the sentence, or the performance will be very bad
    
    # debug for inpt
    # ipdb.set_trace()
    
    inpt = [corpus.en_stoi[word] if corpus.en_stoi.get(word, None) else corpus.en_stoi['<unk>'] for word in sentence.split()]
    inpt[0: 0] = [corpus.en_stoi['<sos>']]
    inpt.append(corpus.en_stoi['<eos>'])
    inpt = torch.LongTensor(inpt)
    inpt = inpt.unsqueeze(0).cuda()    # [1, T] / [B, T]
    

    # model output
    out = seq2seq.eval_forward(inpt, 50, corpus.zh_stoi['<sos>'])

    # decode the out into the chinese sentence
    print('Output:', ' '.join([corpus.zh_itos[word] for word in filter(lambda x: x != corpus.zh_stoi["<eos>"], out)]))
    print('-' * 50)

def show(corpus, train_iter):
    test_batch = random.choice(train_iter)
    src = test_batch.get_src().cuda()
    trg = test_batch.get_target().cuda()
    
    # load dataset
    zh_size, en_size = corpus.zh_size, corpus.en_size
    hidden_size, embed_size = 256, 128
    encoder = Encoder(en_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, zh_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    seq2seq.load_state_dict(torch.load('.save/seq2seq_100.pt'))

    for sent in src.transpose(0, 1):
        print('Input:', ' '.join([corpus.en_itos[word] for word in sent]))
        out = seq2seq.eval_forward(sent.unsqueeze(0), 50, corpus.zh_stoi['<sos>'])
        print('Output:', ' '.join([corpus.zh_itos[word] for word in filter(lambda x: x != corpus.zh_stoi["<eos>"], out)]))
        print('-' * 50)

if __name__ == "__main__":
    args = parse_arguments()
    train, _, _, corpus = load_dataset(args.batch_size)
    
    # TEST
    while True:
        cli(input('Input: '), corpus)

    # test, [T, B, N]
    # show(corpus, train)
