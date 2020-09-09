"""
NMF reconstructs samples
In this exercise, you'll check your understanding of how NMF reconstructs 
samples from its components using the NMF feature values. 
On the right are the components of an NMF model. 
If the NMF feature values of a sample are [2, 1], 
then which of the following is most likely to represent the original sample?

Feature values [2, 1]

NMF components:
[[1.  0.5 0. ]
 [0.2 0.1 2.1]]


Answer:
  2*1   2*0.5  2*0
+ 1*0.2 1*0.1  1*2.1
  -------------------
   2.2   1.1   2.1 

--> [2.2, 1.1, 2.1]
"""