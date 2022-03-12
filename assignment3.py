

import numpy as np
import time
import random
from assignment2 import *

class Assignment3:
    def __init__(self):
        pass
    def intigrate_wrap(self,f,a,b,n):

        h = (b-a)/n
        first = f(a)
        last = f(b)
        x = a
        ssum = 0
        for i in range(n-1):
            x += h
            val = f(x)
            if i % 2 == 0:
                ssum += 4 * val
            else:
                ssum += 2 * val

        total_sum = (h/3) * (first + ssum + last)
        return np.float32(total_sum)

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        if n % 2 == 1:
            return self.intigrate_wrap(f, a, b, n - 1)
        else:
            return self.intigrate_wrap(f, a, b, n - 2)



    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """
        root_finder = Assignment2()
        intersacts = root_finder.intersections(f1, f2, 1, 100)
        intersacts.sort()
        g = lambda x: abs(f1(x) - f2(x))
        if len(intersacts) < 2:
            return np.nan
        sum = 0
        for i in range(len(intersacts) - 1):
            sum += self.integrate(g, float(intersacts[i]), float(intersacts[i + 1]), 52)

        return np.float32(sum)

