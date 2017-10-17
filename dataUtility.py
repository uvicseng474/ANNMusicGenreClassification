#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

class csV2Cla():

    def __init__(self, fname):
        assert(os.path.exists(fname))
        df = pd.read_csv(fname)
        # organize the data by space
        df = df.replace({'\n': ' '}, regex=True)
        df = df.replace({',': ''}, regex=True)
        df = df.replace({'.': ''}, regex=True)
        # remove unhelpful data
        #df.drop(df[df['lyrics'].str.split().str.len() < 10]) # this step made my computer crash don't try
        #self.better_base = df[df['lyrics'].str.split().str.len() >= 10]
        self.base = df
        self.genre = df.genre.unique()
        self.artist = df.artist.unique()
        self.year = df.year.unique()

if __name__ == '__main__':
    cvt = csV2Cla('data/lyrics.csv')
    print(cvt.base.head())
    print(cvt.year)