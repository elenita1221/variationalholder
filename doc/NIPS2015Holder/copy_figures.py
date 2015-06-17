#!/usr/bin/env python
import sys, os

DOC='arxiv-holder/doc/figures/'

f = open(sys.argv[1], 'r')

holder_figures = set(['demo2d.pdf'])

print '\n\n\nRUNNING copy_figures.py\n\n\n'
for line in f:
    line = line.rstrip('\n')
    start = line.find('{') + 1
#    fname = line[start:-1]
    end = line.find('}')
    fname = line[start:end]
    if not fname.endswith('.pdf'):
        fname = fname + '.pdf'
    if fname[:8] == 'figures/':
        fname = fname[8:]
    print fname
    if fname in holder_figures:
        DIR = 'pics/'
    else:
        raise Exception
    cmd = 'cp %s%s %s%s' % (DIR, fname, DOC, fname)
    print cmd
    os.system(cmd)
