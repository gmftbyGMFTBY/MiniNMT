#!/usr/bin/python

'''
Training model script
'''

import os
import math
import argparse
import torch
import ipdb
import time
import random

from torch import optim
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F

from model import Encoder, Decoder, Seq2Seq
from dataset import load_dataset
from vis import Visualizer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=200,
                    help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=64,
                    help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0005,
                    help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                    help='clip the grad to invode the gradient explore')
    return p.parse_args()

def evaluate(model, val_iter, vocab_size, corpus):
    model.eval()
    pad = corpus.en_stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src = batch.get_src().cuda()
        trg = batch.get_target().cuda()

        output = model(src, trg)
        loss = F.cross_entropy(output[1:].view(-1, vocab_size),                                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
                                                                
        total_loss += loss.data[0]
    return total_loss / len(val_iter)

def train(e, model, optimizer, train_iter, vocab_size, grad_clip, corpus, viser, begin):
    model.train()
    total_loss = 0
    pad = corpus.en_stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src = batch.get_src()
        trg = batch.get_target()
        src, trg = src.cuda(), trg.cuda()

        optimizer.zero_grad()
        output = model(src, trg)    # output: [T, B, N], [1:] means do not consider about the <sos>

        loss = F.cross_entropy(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)


        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()    # change into the item to adapt the PyTorch 0.5

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            # plot loss
            t = int(time.time() - begin)
            viser.plot_loss(t, total_loss)
            
            total_loss = 0


def main():
    begin = time.time()
    viser = Visualizer()
    args = parse_arguments()
    hidden_size = 256
    embed_size = 128
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, test_iter, val_iter, corpus = load_dataset(args.batch_size)
    zh_size, en_size = corpus.zh_size, corpus.en_size

    print("[!] Instantiating models...")
    encoder = Encoder(en_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, zh_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)

    print("[!] Begin to train")
    for e in range(1, args.epochs+1):
        train(e, seq2seq, optimizer, train_iter,
              zh_size, args.grad_clip, corpus, viser, begin)
        # val_loss = evaluate(seq2seq, val_iter, zh_size, corpus)
        # print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
        #       % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        # if not best_val_loss or val_loss < best_val_loss:
        if True:
            print("[!] best model occur, saving model...")

            # If find the new best model, try to plot text
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), '.save/seq2seq_%d.pt' % (e))

            # plot text from the test dataset
            # only plot 1 sentences
            test_batch = random.choice(train_iter)
            src = test_batch.get_src()[:, 0].unsqueeze(0).cuda()
            trg = test_batch.get_target()[:, 0].cuda()
            pre = seq2seq.eval_forward(src, 100, corpus.zh_stoi['<sos>'])
            src_sent = ' '.join([corpus.en_itos[int(word)] for word in src[0]])
            trg_sent = ' '.join([corpus.zh_itos[int(word)] for word in trg])
            pre_sent = ' '.join([corpus.zh_itos[int(word)] for word in pre])
            viser.plot_text(e, src_sent, pre_sent, trg_sent)

    # test_loss = evaluate(seq2seq, test_iter, zh_size, corpus)
    # print("[TEST] Final loss:%5.2f" % test_loss)
    print('[!] Training over')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
    except Exception as e:
        print("[ERROR]", e)
