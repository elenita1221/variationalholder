#!/usr/bin/env python
import fileinput


for line in fileinput.input():
    line = line.rstrip('\n')
    if line == r'%':
        # or ('commenting.tex' in line):
        # do not print lines that were commented out
        continue
    if 'includegraphics' not in line:
        print line
    else:
        end = line.find('{') + 1
        start = line.find('/') + 1
        print '%sfigures/%s' % (line[:end], line[start:])
