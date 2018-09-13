#!/usr/bin/python

'''
This module use visdom to show the training process
'''

import ipdb
import visdom
import time
import numpy as np

class Visualizer:
    def __init__(self):
        self.vis = visdom.Visdom(env='seq2seq', port=9999, server='http://localhost')

    def plot_loss(self, x, y):
        self.vis.line(X=[x], Y=[y], win='loss', update='append', opts={'title':'loss'})

    def plot_text(self, idx, src, pre, trg):
        # vis.text(..., append=True, ...) can make the pane show
        # more sentences
        # first text do not use append parameters
        self.vis.text(str(idx) + ' ' + '-' * 20, win='Text')
        self.vis.text('EN: ' + src, append=True, win='Text')
        self.vis.text('ZH_REAL: ' + trg, append=True, win='Text')
        self.vis.text('ZH_PRED: ' + pre, append=True, win='Text')
        

if __name__ == "__main__":
    pass
