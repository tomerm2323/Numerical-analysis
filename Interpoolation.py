"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
import copy




class Interpoolation:
    def __init__(self):
        self. curr_index = 0
        self.dic = {}


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

    def get_bezier_coef(self,points):
        # since the formulas work given that we have n+1 points
        # then n must be this:
        n = len(points) - 1

        # build coefficents matrix
        C = 4 * np.identity(n,dtype=np.float64)
        np.fill_diagonal(C[1:], 1.0)
        np.fill_diagonal(C[:, 1:], 1.0)
        C[0, 0] = 2.0
        C[n - 1, n - 1] = 7.0
        C[n - 1, n - 2] = 2.0

        # build points vector
        P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        P[0] = points[0] + 2 * points[1]
        P[n - 1] = 8 * points[n - 1] + points[n]

        # solve system, find a & b
        A = np.linalg.solve(C, P)
        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 1] = (A[n - 1] + points[n]) / 2

        return A, B


    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. 
        
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

        def get_bezier_cubic2(p1, p2, p3, p4):
            return lambda t: self.get_bezier_cubic(p1, p2, p3, p4, t)

        interpolation_range = list(np.linspace(a, b, n))
        points = [np.asarray((x, f(x)), dtype=np.float64) for x in interpolation_range]
        a_array,b_array = self.get_bezier_coef(points)

        for i in range(0, len(points) - 1):
            self.dic[points[i][0]] = get_bezier_cubic2(points[i],a_array[i],b_array[i], points[i+1])

        return lambda x: np.float64(self.dic[points[self.binary_search(interpolation_range,x)][0]](
            self.t_form(x,points[self.curr_index][0],points[self.curr_index + 1][0])))



