"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
from scipy.optimize import curve_fit
import copy
import math



class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """
        self. curr_index = 0
        self.dic = {}




    def get_dig(self, points):
        n = len(points) - 1
        c = np.c_[np.array([[np.float64(1.0)] * (n - 1)])]
        b = np.c_[2, np.array([[np.float64(4.0)] * (n - 2)]), np.float64(7.0)]
        a = np.c_[np.array([[np.float64(1.0)] * (n - 2)]), np.float64(2.0)]
        return a[0], b[0], c[0]

    def get_p(self, points):
        n = len(points) - 1
        P = [2 * (2 * np.asarray(points[i],dtype=np.float64) + np.asarray(points[i + 1],dtype=np.float64)) for i in range(n)]
        P[0] = np.asarray(points[0]) + 2 * np.asarray(points[n])
        P[n - 1] = 8 * np.asarray(points[1]) + np.asarray(points[n])
        return P

    def tdma2(self,a, b, c, d):
        n = len(b)
        w = np.zeros(n - 1, float)
        g = np.zeros((n, 2), float)
        p = np.zeros((n, 2), float)

        w[0] = c[0] / b[0]
        g[0] = d[0] / b[0]

        for i in range(1, n - 1):
            w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
        for i in range(1, n):
            g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
        p[n - 1] = g[n - 1]
        for i in range(n - 1, 0, -1):
            p[i - 1] = g[i - 1] - w[i - 1] * p[i]
        return p
    # def tdma(self,a, b, c, d):
    #
    #         '''
    #         TDMA solver, a b c d can be NumPy array type or Python list type.
    #         refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    #         '''
    #         nf = len(a)  # number of equations
    #         ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy the array
    #         for it in range(1, nf):
    #             mc = ac[it] / bc[it - 1]
    #             bc[it] = bc[it] - mc * cc[it - 1]
    #             dc[it][0] = dc[it][0] - mc * dc[it - 1][0]
    #
    #         xc = ac
    #         xc[-1] = dc[-1][0] / bc[-1]
    #
    #         for il in range(nf - 2, -1, -1):
    #             xc[il] = (dc[il][0] - cc[il] * xc[il + 1]) / bc[il]
    #
    #         del bc, cc, dc  # delete variables from memory
    #         return xc
    def get_b(self,a_array,points):
        n = len(a_array)
        b = np.zeros((n, 2), float)
        for i in range(n - 1):
            b[i] = 2 * points[i + 1] - a_array[i + 1]
        b[n - 1] = (a_array[n - 1] + points[n]) / 2

        return b

    def binary_search(self, arr, x):
        low = 0
        high = len(arr) - 1
        mid = 0
        if x == arr[-1]:
            return -1
        if x == arr[0]:
            return 0

        while low <= high:

            mid = round((high + low) // 2)

            # If x is greater, ignore left half
            if arr[mid] < x:
                low = mid + 1

            # If x is smaller, ignore right half
            elif arr[mid] > x:
                high = mid - 1

            # means x is present at mid
            else:

                self.curr_index = mid
                return mid

        # If we reach here, then the element was not present

        self.curr_index = high
        return high




    def get_bezier_cubic(self, p1, p2, p3, p4, t):
        return (np.power(1 - t, 3) * p1 + 3 * np.power(1 - t, 2) * t * p2 +
                             3 * (1 - t) * np.power(t,2) * p3 + np.power(t, 3) * p4)[1]


    def t_form (self, x, a, b):
        return (x - a) / (b - a)



    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # def get_bezier_cubic2(p1, p2, p3, p4):
        #     return lambda t: (np.power(1 - t, 3) * p1 + 3 * np.power(1 - t, 2) * t * p2 +
        #                       3 * (1 - t) * np.power(t, 2) * p3 + np.power(t, 3) * p4)[1]

        def get_bezier_cubic2(p1, p2, p3, p4):
            return lambda t: self.get_bezier_cubic(p1, p2, p3, p4, t)

        interpolation_range = list(np.linspace(a, b, n))
        points = [np.asarray((x, f(x)), dtype=np.float64) for x in interpolation_range]
        ad, bd, cd = self.get_dig(points)
        p_array = self.get_p(points)

        a_array = self.tdma2(ad, bd, cd, d=p_array)

        b_array = self.get_b(a_array, points)

        for i in range(len(points) - 1):
            self.dic[points[i][0]] = get_bezier_cubic2(points[i],a_array[i],b_array[i], points[i+1])

        ###############################################################################
        # flag = False
        #
        # for i in range(0,len(points)):
        #     if not flag:
        #
        #         if len(points[i:-1]) >= 2:
        #             self.dic[points[i][0]] = bezier2(points[i],points[i + 1],points[i + 2])
        #
        #         else:
        #             for j in range(i, len(points), 1):
        #                 self.dic[points[j][0]] = bezier2(points[j - 2], points[j-1], points[j])
        #
        #             flag = True
        #
        #

        return lambda x: np.float64(self.dic[points[self.binary_search(interpolation_range,x)][0]](
            self.t_form(x,points[self.curr_index][0],points[self.curr_index + 1][0])))



##########################################################################



import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print('test 30 poly: ')
        print("Run Time: {time}".format(time=T))
        print("Error: {err}".format(err=mean_err))
    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

    #

    def test_sin(self):
        T = time.time()
        ass1 = Assignment1()

        mean_err = 0
        for i in tqdm(range(100)):

            f = lambda x: np.sin(x**2)
            ff = ass1.interpolate(f, -10,10, 100)



            xs = np.random.uniform(0,10,200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)
            err = err / 200
            mean_err += err
        mean_err = mean_err / 100



        T = time.time() - T
        print('test sin: ')
        print("Run Time: {time}".format(time=T))
        print("Error: {err}".format(err=mean_err))

    def test_abs(self):
        T = time.time()
        ass1 = Assignment1()

        mean_err = 0
        for i in tqdm(range(100)):

            f = lambda x: abs(x)*i

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print('test abs: ')
        print("Run Time: {time}".format(time=T))
        print("Error: {err}".format(err=mean_err))

    def test_ratio(self):
        T = time.time()
        ass1 = Assignment1()

        mean_err = 0
        for i in tqdm(range(100)):

            f = lambda x: abs(np.sin(2*i*x))
            ff = ass1.interpolate(f, -100, 100, 10)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print('test ratio: ')
        print("Run Time: {time}".format(time=T))
        print("Error: {err}".format(err=mean_err))



if __name__ == "__main__":
    unittest.main()



