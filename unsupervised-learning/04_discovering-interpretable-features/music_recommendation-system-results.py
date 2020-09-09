"""
Recommend musical artists part II
Suppose you were a big fan of Bruce Springsteen - which other musicial artists might you like? Use your NMF features from the previous exercise and the cosine similarity to find similar musical artists. A solution to the previous exercise has been run, so norm_features is an array containing the normalized NMF features as rows. The names of the musical artists are available as the list artist_names.

Instructions

Import pandas as pd.
Create a DataFrame df from norm_features, using artist_names as an index.
Use the .loc[] accessor of df to select the row of 'Bruce Springsteen'. Assign the result to artist.
Apply the .dot() method of df to artist to calculate the dot product of every row with artist. Save the result as similarities.
Print the result of the .nlargest() method of similarities to display the artists most similar to 'Bruce Springsteen'.
"""
# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities)
print("\nYOUR RECOMMENDATIONS")
print(similarities.nlargest())

"""

<script.py> output:
    Massive Attack           0.000000
    Sublime                  0.000000
    Beastie Boys             0.140255
    Neil Young               0.955896
    Dead Kennedys            0.067893
                               ...   
    Franz Ferdinand          0.000571
    The Postal Service       0.000723
    The Dresden Dolls        0.021921
    The Killers              0.057398
    Death From Above 1979    0.008020
    Length: 111, dtype: float64
    
    YOUR RECOMMENDATIONS
    Bruce Springsteen    1.000000
    Neil Young           0.955896
    Van Morrison         0.872452
    Leonard Cohen        0.864763
    Bob Dylan            0.859047
    dtype: float64
"""