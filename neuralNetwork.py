from dataUtility import csV2Cla
import os.path

class NeuralNetwork():
    
    def __init__(self):
        print('todo')
        # store helpful data structures

    def train(self, X, y):
        print('todo')
        # Binary input patterns
        # For a set of binary patterns s(p), p = 1 to P
        # Here, s(p) = s1(p), s2(p),…, si(p),…, sn(p)
        # Weight Matrix is given by
        # 𝐰𝐢𝐣 = ∑ [𝟐𝐬𝐢(𝐩) − 𝟏][𝟐𝐬𝐣(𝐩) − 𝟏 ] 𝐟𝐨𝐫 𝐢 ≠ j (∑ goes from 1 to P)
        
    def test(self, X, y):
        print('todo')


if not os.path.exists('data/results.pickle'):
    data = csV2Cla('data/lyrics.csv','data/result.pickle')
else:
    data = csV2Cla()
    data.load('data/result.pickle')

# run python neuralNetwork.py
# below statements are to understand data structure
print("List of genres \n", data.genre)
print("\nList of years :\n", data.year)
print("\nList of artists :\n", data.artist)
print("\nDescription of data :\n", data.base.head())
print("\nList of lyrics :\n", data.base.lyrics)