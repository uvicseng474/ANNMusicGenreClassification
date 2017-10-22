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
        df.lyrics = df.lyrics.str.replace(',', '')
        df.lyrics = df.lyrics.str.replace('.', '')
        df.lyrics = df.lyrics.str.replace('\[.*\]', '')
        df.lyrics = df.lyrics.str.replace(':', '')
        df.lyrics = df.lyrics.str.replace('(', '')
        df.lyrics = df.lyrics.str.replace(')', '')
        df.lyrics = df.lyrics.str.replace('!', '')
        df = df[df.lyrics != "INSTRUMENTAL"]
        #split the strings during processing
        #doing it all at once will take too long
        #we can limit the time taken during processing
        self.base = df
        self.genre = df.genre.unique()
        self.artist = df.artist.unique()
        self.year = df.year.unique()

if __name__ == '__main__':
    cvt = csV2Cla('data/lyrics.csv')
    print(cvt.base.head())
    print(cvt.year)
    print(cvt.base.lyrics)