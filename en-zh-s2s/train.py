import os, math, argparse
import torch, ipdb
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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


def evaluate(model, val_iter, vocab_size, corpus):
    model.eval()
    pad = corpus.en_stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src = batch.get_src().cuda()
        trg = batch.get_target().cuda()

        output = model(src, trg)
        loss = F.cross_entropy(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        
        total_loss += loss.data[0]
    return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, corpus):
    # ipdb.set_trace()
    model.train()
    total_loss = 0
    pad = corpus.en_stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src = batch.get_src()
        trg = batch.get_target()
        src, trg = src.cuda(), trg.cuda()

        optimizer.zero_grad()
        output = model(src, trg)    # output: [T, B, N], [1:] means do not consider about the <sos>

        ipdb.set_trace()
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
            total_loss = 0

def test_by_human():
    # test the result from .de to .en
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset for test ...")
    train_iter, val_iter, test_iter, corpus = load_dataset(args.batch_size)
    zh_size, en_size = corpus.zh_size, corpus.en_size

    # load the model
    encoder = Encoder(en_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, zh_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    seq2seq.load_state_dict(torch.load('.save/seq2seq_21.pt'))

    # only decoder 1 batch sents
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        # do not use force teaching, just use the maximum possibility
        output = seq2seq(src, trg, 0)
        output = output.transpose(0, 1)    # (B*T*N)
        src = src.transpose(0, 1)          # (B*T)
        
        # src
        for source, result in zip(src, output):
            print('German: : ')
            print('    ', end=' ')
            for word in source:
                if DE.vocab.itos[word] in ["<pad>", "<sos>", "<unk>", "<eos>"]:
                    continue
                print(DE.vocab.itos[word], end=' ')
                # print(word, end=' ')
            print()
            print('English: ')
            print('    ', end=' ')
            for word in result:
                _, index = word.max(0)
                if EN.vocab.itos[index] in ["<pad>", "<sos>", "<unk>", "<eos>"]:
                    continue
                print(EN.vocab.itos[index], end=' ')
                # print('test ...', word)
            print()

        print("[!] End the testing ...")
        break

def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
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
    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(e, seq2seq, optimizer, train_iter,
              zh_size, args.grad_clip, corpus)
        val_loss = evaluate(seq2seq, val_iter, zh_size, corpus)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] best model occur, saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), '.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss

    test_loss = evaluate(seq2seq, test_iter, zh_size, corpus)
    print("[TEST] Final loss:%5.2f" % test_loss)

if __name__ == "__main__":
    try:
        # main()
        test_by_human()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
