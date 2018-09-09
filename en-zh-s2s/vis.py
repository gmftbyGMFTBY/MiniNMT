#!/usr/bin/python

'''
This module use visdom to show the training process
'''

import visdom
import time
import numpy as np

class Visualizer:
    def __init__(self):
        self.vis = visdom.Visdom(env='seq2seq')

    def plot_loss(self, x, y):
        self.vis.line(X=np.array([x]), Y=np.array([y]), win='loss', update='append')

    def plot_text(self, src, pre, trg):
        # vis.text(..., append=True, ...) can make the pane show
        # more sentences
        self.vis.text('-' * 20, append=True, win='Text')
        self.vis.text('EN: ' + src, append=True, win='Text')
        self.vis.text('ZH_REAL: ' + trg, append=True, win='Text')
        self.vis.text('ZH_PRED: ' + pre, append=True, win='Text')
        

if __name__ == "__main__":
    pass