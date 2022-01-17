"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random
import math


class Assignment4A:
    def __init__(self):
        pass


    def create_A(self,deg, fit_range,f):
        A = []
        for x in fit_range:
            A.append(np.asarray(list(map(lambda power: x**power, range(deg,-1,-1)))))
        return np.asarray(A)
    def create_poly(self,coeff):
        pol = np.poly1d(coeff)
        return pol

    def avgy(self,func,x):
        x_lst = [x] * 50
        y_lst = list(map(lambda x: func(x), x_lst))
        #y_lst = func(x_lst)
        avg = np.average(y_lst)
        return avg





    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        fit_range = np.linspace(a, b, 1000, dtype=np.float64)
        A9 = np.vander(fit_range, d + 1)
        A = self.create_A(d,fit_range,f)
        # AT = A.T
        y = np.asarray(list(map(lambda x: self.avgy(f, x), fit_range)), dtype=np.float64)
        AT = A.T
        ATA = np.dot(A.T, A)
        ATA_inv = np.linalg.inv(ATA)
        ATA_invAT = np.dot(ATA_inv, AT)
        coeff = np.dot(ATA_invAT, y)
        my_fit = self.create_poly(coeff)

        return my_fit


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(2, 1, 1, 1, 1, 6, 1, 1, 3, 9, 13, 5, 3)
        nf = NOISY(1)(f)
        print()
        print(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=-5, b=5, d=12, maxtime=5)
        print()
        print(ff)
        T = time.time() - T
        mse=0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x)-ff(x))**2
        mse = mse/1000
        print('MSE: ' )
        print(mse)


    def test_lin(self):
        f = poly(42,-1463)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=7, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x)-ff(x))**2
        mse = mse/1000
        print(T)
        print(mse)

        



if __name__ == "__main__":
    unittest.main()
