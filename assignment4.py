import numpy as np
import time
import random


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
# The assumption is that func is noisy
    def avgy(self,func,x,maxtime, startime): 
        if time.time() - startime >= 0.97 * maxtime:
            raise StopIteration()
        x_lst = [x] * 50
        y_lst = list(map(lambda x: func(x), x_lst))
        #y_lst = func(x_lst)
        avg = np.average(y_lst)
        return avg

    def invert_matrix(self,matrix):
        inv_matrix = []
        Isize = len(matrix)
        I = np.identity(Isize)
        IT = I.T
        for i in range(Isize):
            inv_matrix.append(np.linalg.solve(matrix,IT[i].T))
        return inv_matrix


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
        start = time.time()
        fit_range = np.linspace(a, b, 1000, dtype=np.float64)
        A = self.create_A(d,fit_range,f)
        y = np.asarray(list(map(lambda x: self.avgy(f, x, maxtime, start), fit_range)), dtype=np.float64)
        if len(y) < 1000:
            coeff = np.random.uniform(-1,5, d + 1)
            poly = self.create_poly(coeff)
            return poly
        AT = A.T
        ATA = np.dot(A.T, A)
        ATA_inv = self.invert_matrix(ATA)
        ATA_invAT = np.dot(ATA_inv, AT)
        coeff = np.dot(ATA_invAT, y)
        my_fit = self.create_poly(coeff)

        return my_fit
