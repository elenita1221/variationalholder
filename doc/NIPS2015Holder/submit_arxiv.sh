#!/usr/bin/env bash

# script for submitting holder paper to arxiv
rm -rf arxiv-holder
DOC='arxiv-holder/doc'
mkdir arxiv-holder $DOC
mkdir $DOC/figures
cp holder.tex arxiv-holder.tex
cp holder.bbl arxiv-holder.bbl
# recent version has been stripped of all comments
perl -pe 's/(^|[^\\])%.*/\1%/' < arxiv-holder.tex | ./fix_paths.py > $DOC/arxiv-holder.tex
cp holder.bib $DOC   # bib file required?
cp arxiv-holder.bbl $DOC
grep includegraphics $DOC/arxiv-holder.tex > figures.txt
# IMPORTANT: edit directory in copy_figures.py
./copy_figures.py figures.txt
cd arxiv-holder
zip -r doc.zip doc
# commented out following line after adding arxiv-holder.zip to svn
# cp doc.zip ../arxiv-holder.zip
cd ..


