"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from sympy import limit, Symbol
from functionUtils import AbstractShape
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
import copy
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.cluster import KMeans
from scipy import *
from scipy.misc import derivative

class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self,area):

        self.a = area
    def area(self):
        return self.a


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        points = [contour() for i in range(10000)]
        hull = ConvexHull(points)
        area = np.float32(hull.area / 2)
        return area


    #
    def distances(self,point):
        x = point[0]
        y = point[1]
        d = np.sqrt(x**2 + y**2)
        return d
    def plot(self,points):
        x_lst = []
        y_lst = []
        for p in points:
            x_lst.append(p[0])
            y_lst.append(p[1])
        plt.scatter(x_lst,y_lst)
        plt.show()
    def fit_line(self,point1,point2):
        x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
        param = np.polyfit([x1, x2], [y1, y2], 1)
        a = param[0]
        b = param[1]
        return lambda x: a * x + b

    def get_coords(self, points):
        x_lst = []
        y_lst = []
        for p in points:
            x_lst.append(p[0])
            y_lst.append(p[1])
        return x_lst, y_lst

    def k_means(self,points, k):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
        return kmeans.cluster_centers_

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """

        #replace these lines with your solution

        T = time.time()
        area_lst =[]
        a_list = []
        num_of_means = []
        samples = [sample() for i in range(4000)]
        self.plot(samples)
        for i in range(3, 35):
            if time.time() - T > 0.95 * maxtime:
                return MyShape(np.average(area_lst))
            num_of_means.append(i)
            centers = list(self.k_means(samples,k=i))
            hull = ConvexHull(centers)
            a = np.float64(hull.area/2)
            area_lst.append(a)
        coefs = np.polyfit(num_of_means, area_lst,3)
        def fited_func(x):
            return coefs[0] * x**3 + coefs[1] *x**2 + coefs[2] * x + coefs[3]
        dev_lst = []
        for i in range(3,35):
            dev_at_point = derivative(fited_func, i, dx=1e-6)
            dev_lst.append(dev_at_point)
        sweet_spot = min(dev_lst)
        sweet_spot_index = dev_lst.index(sweet_spot)
        k_list = []
        for i in range(sweet_spot_index - 2, sweet_spot_index + 2):
            area = area_lst[i]
            k = i
            a_list.append(area)
            k_list.append(k)
        coefs = np.polyfit(k_list, a_list, 3)
        poly = np.poly1d(coefs)
        samples_range = np.linspace(k_list[0],k_list[-1],100)
        sumofsampels = 0
        for s in samples_range:
            sumofsampels += poly(s)
        final_area = sumofsampels/100
        result = MyShape(final_area)
        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(1)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        a1 = ass5.area(circ)
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
