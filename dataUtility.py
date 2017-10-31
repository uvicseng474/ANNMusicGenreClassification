#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import pickle

from collections import Counter

class csV2Cla():

    def __init__(self, fname = None, fout = None, fpickle = None, fcount = None):
        print('Reading and filtering data... ', end='')
        if fname is None:
            return
        assert(os.path.exists(fname))
        df = pd.read_csv(fname, low_memory=False)
        # remove unhelpful data
        df = df.dropna()
        df.lyrics = df.lyrics.str.replace('\n', ' ')
        df.lyrics = df.lyrics.str.replace(r"[\-\"',.()!?:;]|(\[.*\])", '')
        df.lyrics = df.lyrics.str.lower()
        df.lyrics = df.lyrics.str.strip()
        # Delete records where lyrics match the list below after above filters are applied
        disallow_lyrics = ["instrumental", "instru", ""]
        for da in disallow_lyrics:
            df = df[df.lyrics != da]
        df = df[df.genre != "Not Available"]
        # 'Other' genre is mainly composed of other languages
        df = df[df.genre != "Other"]
        print('Done')
        if(fout is not None and fpickle is not None and fcount is not None):
            print('Save data frame to csv...', end='')
            df.to_csv(fout, encoding="utf8")
            print('Done')
        print('Finding unique values... ', end='')
        self.base = df
        self.genre = df.genre.unique()
        self.artist = df.artist.unique()
        self.year = df.year.unique()
        self.count_all = Counter(" ".join(df.lyrics.values.tolist()).split(" "))
        #self.lyrics_vector = df.lyrics.str.split(' ')
        print('Done')
        if(fpickle is not None and fcount is not None):
            print('Save data to pickle...', end='')
            f = open(fpickle, 'wb')
            fc = open(fcount, 'wb')
            pickle.dump([" ".join(str(v) for v in self.genre), " ".join(str(v) for v in self.artist), " ".join(str(v) for v in self.year)], f)
            pickle.dump(self.count_all, fc)
            print('Done')

    def load(self, fin, fpickle, fcount):
        # check paths
        print('Reading data... ', end='')
        assert(os.path.exists(fin))
        assert(os.path.exists(fpickle))
        assert(os.path.exists(fcount))
        self.base = pd.read_csv(fin, low_memory=False, encoding = "utf8")
         # get back pickle objects
        with open(fpickle, 'rb') as f:
            sgenre, sartist, syear = pickle.load(f)
            self.genre = sgenre.split(' ')
            self.artist = sartist.split(' ')
            self.year = syear.split(' ')
        with open(fcount, 'rb') as f:
            self.count_all = pickle.load(f)
        print('Done')

    def getFreqWords(self, arr, n):
        x = np.concatenate([np.array(i) for i in arr])
        print(x[1])
        words = np.unique(x, return_counts=True)
        words = np.array(words, dtype=object).T
        words = words[np.argsort(words[:,1])[::-1][:n]]
        return words

    def getFreqWordsByGenre(self, genre):
        print('Getting most frequent words in genre: ' + genre)
        return Counter(" ".join(self.base[self.base.genre == genre].lyrics.values.tolist()).split(" "))

    def featureExtraction(self, n, max):
        print('Extracting ...')
        most_common_by_genre = {}
        for genre in cvt.genre:
            genredict = cvt.getFreqWordsByGenre(genre)
            most_common_by_genre.update({genre: Counter(dict(genredict.most_common(n)))})

        common_words = []
        for cw in most_common_by_genre.values():
            for w in list(cw):
                common_words.append(w)
        all_genres = Counter(common_words)
        remove = [w for w in list(all_genres) if all_genres[w]>2]

        for genre,cw in most_common_by_genre.items():
            for w in list(cw):
                if w in remove:
                    del cw[w]
            most_common_by_genre[genre] = [wtuple[0] for wtuple in cw.most_common(max)]
        return most_common_by_genre

if __name__ == '__main__':
    cvt = csV2Cla('data/lyrics.csv', 'data/results.csv', 'data/result.pickle','data/counts.pickle')

    # print(cvt.base.head())
    # print(cvt.year)
    # print(cvt.genre)
    # print(cvt.base.lyrics)


    # Considering top 200 most common words from each genre and deleting words that appear
    # in 3 or more genres' list yields a small list of feature words for each genre
    # Note: some non-english words are still making it into the pop genre
    print(cvt.featureExtraction(200, 30))
