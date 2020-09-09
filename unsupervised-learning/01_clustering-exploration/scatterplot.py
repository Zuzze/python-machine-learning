"""
How many clusters?
You are given an array points of size 300x2, where each row gives the (x, y) co-ordinates of a point on a map. Make a scatter plot of these points, and use the scatter plot to guess how many clusters there are.

matplotlib.pyplot has already been imported as plt. In the IPython Shell:

Create an array called xs that contains the values of points[:,0] - that is, column 0 of points.
Create an array called ys that contains the values of points[:,1] - that is, column 1 of points.
Make a scatter plot by passing xs and ys to the plt.scatter() function.
Call the plt.show() function to show your plot.
How many clusters do you see?


In [1]: xs = points[:, 0]

In [2]: ys = points[:, 1]

In [3]: plt.scatter(xs, ys)
Out[3]: <matplotlib.collections.PathCollection at 0x7f92d9f5d588>

In [4]: plt.show()


Answer: 3 clusters

"""

