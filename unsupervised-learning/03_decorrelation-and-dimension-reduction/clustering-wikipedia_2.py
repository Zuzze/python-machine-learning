"""
Exercise
Clustering Wikipedia part II
It is now time to put your pipeline from the previous exercise to work! You are given an array articles of tf-idf word-frequencies of some popular Wikipedia articles, and a list titles of their titles. Use your pipeline to cluster the Wikipedia articles.

A solution to the previous exercise has been pre-loaded for you, so a Pipeline pipeline chaining TruncatedSVD with KMeans is available.

Instructions
100 XP
Import pandas as pd.
Fit the pipeline to the word-frequency array articles.
Predict the cluster labels.
Align the cluster labels with the list titles of article titles by creating a DataFrame df with labels and titles as columns. This has been done for you.
Use the .sort_values() method of df to sort the DataFrame by the 'label' column, and print the result.
Hit 'Submit Answer' and take a moment to investigate your amazing clustering of Wikipedia pages!
"""
# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))


"""
    label                                        article
59      0                                    Adam Levine
57      0                          Red Hot Chili Peppers
56      0                                       Skrillex
55      0                                  Black Sabbath
54      0                                 Arctic Monkeys
53      0                                   Stevie Nicks
52      0                                     The Wanted
51      0                                     Nate Ruess
50      0                                   Chad Kroeger
58      0                                         Sepsis
0       1                                       HTTP 404
6       1                    Hypertext Transfer Protocol
9       1                                       LinkedIn
8       1                                        Firefox
7       1                                  Social search
5       1                                         Tumblr
4       1                                  Google Search
3       1                                    HTTP cookie
2       1                              Internet Explorer
1       1                                 Alexa Internet
10      2                                 Global warming
11      2       Nationally Appropriate Mitigation Action
18      2  2010 United Nations Climate Change Conference
17      2  Greenhouse gas emissions by the United States
16      2                                        350.org
15      2                                 Kyoto Protocol
14      2                                 Climate change
13      2                               Connie Hedegaard
19      2  2007 United Nations Climate Change Conference
12      2                                   Nigel Lawson
42      3                                    Doxycycline
43      3                                       Leukemia
44      3                                           Gout
48      3                                     Gabapentin
46      3                                     Prednisone
47      3                                          Fever
49      3                                       Lymphoma
41      3                                    Hepatitis B
45      3                                    Hepatitis C
40      3                                    Tonsillitis
35      4                Colombia national football team
38      4                                         Neymar
37      4                                       Football
36      4              2014 FIFA World Cup qualification
34      4                             Zlatan Ibrahimović
33      4                                 Radamel Falcao
32      4                                   Arsenal F.C.
31      4                              Cristiano Ronaldo
30      4                  France national football team
39      4                                  Franck Ribéry
20      5                                 Angelina Jolie
21      5                             Michael Fassbender
22      5                              Denzel Washington
23      5                           Catherine Zeta-Jones
27      5                                 Dakota Fanning
25      5                                  Russell Crowe
26      5                                     Mila Kunis
28      5                                  Anne Hathaway
24      5                                   Jessica Biel
29      5                               Jennifer Aniston
"""