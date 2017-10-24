from dataUtility import csV2Cla

class NeuralNetwork():
    
    def __init__(self):
        # store helpful data structures

    def train(self, X, y):
        # Binary input patterns
        # For a set of binary patterns s(p), p = 1 to P
        # Here, s(p) = s1(p), s2(p),…, si(p),…, sn(p)
        # Weight Matrix is given by
        # 𝐰𝐢𝐣 = ∑ [𝟐𝐬𝐢(𝐩) − 𝟏][𝟐𝐬𝐣(𝐩) − 𝟏 ] 𝐟𝐨𝐫 𝐢 ≠ j (∑ goes from 1 to P)
        
    def test(self, X, y):

data =  csV2Cla('data/lyrics.csv')

# run python neuralNetwork.py
# below statements are to understand data structure
print("List of genres \n", data.genre)
print("\nList of years :\n", data.year)
print("\nList of artists :\n", data.artist)
print("\nDescription of data :\n", data.base.head())
print("\nList of lyrics :\n", data.base.lyrics)