"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def dev_at_point(self, func, point):
        dx = 0.0000001
        df = func(point) - func(point - dx)
        return round(df/(dx))

    def newton_raphson(self, g, guss, delta, lower_range_limit,upper_range_limit):
        z = guss
        iterations = 0
        # largest_x_visited = guss
        while abs(g(z)) > delta and iterations < 100:
            dev = self.dev_at_point(g, z)
            if dev == 0:
                return None
            z = z - g(z)/dev
            iterations += 1

            # if z > largest_x_visited:
            #     largest_x_visited = z

        if z > upper_range_limit or z < lower_range_limit or abs(g(z)) > delta:
            return None
        return z
    # def redundent(self,points):
    #     points.sort()
    #     for i in range(len(points)):
    #         if round(points[i]) == round(points[i + 1]) and self.dev_at_point(func, points[i])*self.dev_at_point(func,points[i+1]) > 0:
    #             doubles.append(roots[i+1])
    #         if points[i + 1] == points[-1]:
    #             break



    def no_double(self, roots, func):
        if len(roots) > 0:
            roots.append(roots[0])
        else:
            return []
        doubles =[]
        roots.sort()
        for i in range(len(roots)):
            if round(roots[i]) == round(roots[i + 1]) and self.dev_at_point(func, roots[i])*self.dev_at_point(func, roots[i+1]) > 0:
                doubles.append(roots[i+1])
            if roots[i + 1] == roots[-1]:
                break


        for root in doubles:
            roots.remove(root)
        return roots
    # def guss(self,curr_point,func,upper_bound):
    #
    #     guss = curr_point
    #     iteration = 0
    #     while self.dev_at_point(point=guss,func=func) * self.dev_at_point(point=curr_point,func=func) > 0:
    #         guss = np.random.uniform(curr_point, upper_bound + 0.01)
    #         iteration += 1
    #         if iteration == 40:
    #             break
    #
    #     return guss



    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """



        g = lambda x: f1(x) - f2(x)
        roots = []
        guss = a
        points = list(np.random.uniform(a,b, 2500 + round(b-a)))
        points = self.no_double(points, g)

        for point in points:

            root = self.newton_raphson(g, point, maxerr, a, b)
            if root is not None:
                roots.append(root)
            # else:
            #     if len(roots) > 0:
            #         guss = roots[0]
            #
            #     else:
            #         guss = b
            #
            # guss = self.guss(guss, g, b)
        roots = self.no_double(roots, g)

        return list(set(roots))


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):



    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print('Test1 roots: ')
        print(X)

    def test_poly(self):
        t = time.time()
        ass2 = Assignment2()
        f1 = lambda x:  8*np.sin(5*x) -np.cos(x/7)*x + x + 5
        f2 = lambda x: 0

        X = ass2.intersections(f1, f2, -10, 44, maxerr=0.005)

        for x in X:
            self.assertGreaterEqual(0.005, abs(f1(x) - f2(x)))
        t = time.time() - t
        print('Test2 roots: ')
        print(X)
        print('Run Time: ')
        print(t)

    def test_sin(self):
        t = time.time()
        ass2 = Assignment2()
        f1 = lambda x: np.sin(100 * x)
        f2 = lambda x: x

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        print(len(X))
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        t = time.time() - t
        print('Test3 roots: ')
        print(X)
        print('Run Time: ')
        print(t)

    def test_cos(self):
        t = time.time()
        ass2 = Assignment2()
        f1 = lambda x: np.cos(x)
        f2 = lambda x: np.sin(x)

        X = ass2.intersections(f1, f2, -30, 30, maxerr=0.001)
        print(len(X))
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        t = time.time() - t
        print('Test3 roots: ')
        print(X)
        print('Run Time: ')
        print(t)


if __name__ == "__main__":
    unittest.main()
