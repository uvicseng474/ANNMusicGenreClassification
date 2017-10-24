#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

class csV2Cla():

    def __init__(self, fname):
        assert(os.path.exists(fname))
        df = pd.read_csv(fname, low_memory=False)
        # remove unhelpful data
        df = df.dropna()
        df.lyrics = df.lyrics.str.replace('\n', ' ')
        df.lyrics = df.lyrics.str.replace(r"[',.()!?:]|(\[.*\])", '')
        df = df[df.lyrics != "INSTRUMENTAL"]
        df = df[df.genre != "Not Available"]
        #split the strings during processing
        #doing it all at once will take too long
        #we can limit the time taken during processing
        df['word_count'] = df.lyrics.str.split().str.len()
        self.base = df
        self.genre = df.genre.unique()
        self.artist = df.artist.unique()
        self.year = df.year.unique()
        #results = set()  useful method to create a set from a very large dataframe
        self.lyrics_vector = df.lyrics.str.split(' ')#.apply(results.update)
        #self.result = results

    def getFreqWords(self, i, n):
        words = np.unique(self.lyrics_vector[1], return_counts=True)
        words = np.array(words, dtype=object).T
        words = words[np.argsort(words[:,1])[::-1][:n]]
        return words

if __name__ == '__main__':
    cvt = csV2Cla('data/lyrics.csv')

    print(cvt.base.head())
    print(cvt.year)
    print(cvt.genre)
    print(cvt.base.lyrics)

    i = 1
    print(cvt.lyrics_vector[i])
    print('-'*10)
    cwords = cvt.getFreqWords(i, 10)
    print('Top 10 Most Common Words:\n{}'.format(cwords))
    print('-'*10)
    #print(cvt.result)
