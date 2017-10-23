from dataUtility import csV2Cla

data =  csV2Cla('data/lyrics.csv')

# run python neuralNetwork.py
# below statements are to understand data structure
print("List of genres \n", data.genre)
print("\nList of years :\n", data.year)
print("\nList of artists :\n", data.artist)
print("\nDescription of data :\n", data.base.head())
print("\nList of lyrics :\n", data.base.lyrics)