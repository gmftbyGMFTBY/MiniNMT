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

def cli(sentence):
    # command line to transform the input English sentence
    args = parse_arguments()
    hidden_size, embed_size = 512, 256
    assert torch.cuda.is_avaiable()

    train_iter, val_iter, test_iter, corpus = load_dataset(args.batch_size)
    zh_size, en_size = corpus.zh_size, corpus.en_size

    # load dataset
    encoder = Encoder(en_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, zh_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    seq2seq.load_state_dict(torch.load('.save/seq2seq_21.pt'))

    # convert the sentence into the batch input to get the output
    inpt = torch.Tensor([corpus.en_stoi[word] for word in sentence.split()])
    inpt = inpt.unsqueeze(0).cuda()    # [1, T] / [B, T]

    pass



if __name__ == "__main__":
    pass