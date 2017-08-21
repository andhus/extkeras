from __future__ import division, print_function

from os import path
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

char_sequence = 'abdcbedbacdcbeabcbe'
vocab = list(set(char_sequence))
char_to_int = dict(zip(vocab, range(len(vocab))))
f_size = (15, 9)
savepath = '/home/andershuss/Data'
sequence = np.zeros((len(char_sequence), len(vocab)))
for t, c in enumerate(char_sequence):
    sequence[t, char_to_int[c]] = 1.

f, axs = plt.subplots(2, 1, sharex=True)

axs[0].imshow(sequence.T, aspect='auto')

x = np.arange(len(char_sequence))
x_plot = np.arange(-0.5, len(char_sequence) - 0.5, 0.1)
mus = [10, 11, 13]
sigmas = [4, 1.5, 2]
pdfs = [norm.pdf(x, mu, sigma) for mu, sigma in zip(mus, sigmas)]
pdfs_plot = [norm.pdf(x_plot, mu, sigma) for mu, sigma in zip(mus, sigmas)]
max_ = max([pdf.max() for pdf in pdfs_plot])

for pdf, c in zip(pdfs_plot, ['b', 'y', 'g']):
    axs[1].plot(x_plot, pdf)

axs[0].set_xlim(-0.5, len(sequence) - 0.5)
axs[1].set_ylim(-0.05 * max_, 1.05 * max_)
f.set_size_inches(*f_size)
f.savefig(path.join(savepath, 'seq_att_byg.png'))


ws = [pdf[:, None] * sequence for pdf in pdfs]
fs = []
for w, pdf, c in zip(ws, pdfs_plot, ['b', 'y', 'g']):
    f_, axs_ = plt.subplots(2, 1, sharex=True)
    axs_[0].matshow(w.T, aspect='auto')
    axs_[1].plot(x_plot, pdf, c=c)
    axs_[0].set_xlim(-0.5, len(sequence) - 0.5)
    axs_[1].set_ylim(-0.05 * max_, 1.05 * max_)
    fs.append(f_)
    f_.set_size_inches(*f_size)
    f_.savefig(path.join(savepath, 'seq_att_{}.png'.format(c)))
