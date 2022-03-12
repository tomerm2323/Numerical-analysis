import numpy as np
import time
import random
from collections.abc import Iterable
from scipy.optimize import bisect

class intersections:
    def __init__(self):
        pass

    import numpy as np
    def gus_for_bisection(self, funcion, point):
        for i in range(100):
            b1 = point + i/(i + 5)
            b2 = point - i/(i + 5)
            if funcion(point) * funcion(b1) < 0:
                return b1
            elif funcion(point) * funcion(b2) < 0:
                return b2

        return None

    def secant(self,f, x0, x1, maxerr):
        fx0 = f(x0)
        try:
            fx1 = f(x1)
        except:
            return None
        iter = 0
        while abs(fx1) > maxerr:
            if iter == 100:
                return None
            iter += 1

            # do calculation
            deltay = fx1 - fx0
            if deltay != 0:
                x2 = (x0 * fx1 - x1 * fx0) / (fx1 - fx0)
            # shift variables (prepare for next loop)
                x0, x1 = x1, x2
                fx0 = fx1
                try:
                    fx1 = f(x2)
                except:
                    return None

            else:
                return None

        return x1


    def dev_at_point(self, func, point):
        dx = 0.0000001
        df = np.float64(func(point) - func(point - dx))
        return round(df/(dx))

    def newton_raphson(self, g, guss, delta, lower_range_limit,upper_range_limit):
        z = guss
        iterations = 0
        
        while abs(g(z)) > delta and iterations < 100:
            dev = self.dev_at_point(g, z)
            if dev == 0:
                root = self.secant(g, x0=z, x1=0.5 + z, maxerr=delta)
                if root is None or root < lower_range_limit or root > upper_range_limit:
                    root = None
                return root
            z = z - g(z)/dev
            iterations += 1

        if z > upper_range_limit or z < lower_range_limit or abs(g(z)) > delta:
            return None
        return z

    def select_roots(self, roots, func, maxerr):

        roots.sort()
        real_roots = [roots[0]]
        compering_point = roots[0]
        
        for i in range(1,len(roots)):
            x1 = compering_point
            x2 = roots[i]
            deltaX = abs(x1 - x2) ## |Xcompre - Xi|
            y1 = func(compering_point)
            y2 = func(roots[i])
            deltaY = y1 is not None and y2 is not None and abs(y1 - y2) #|f(Xcompre) - f(Xi)|
            
            if deltaX > maxerr and deltaY < maxerr:  # Different points have different roots
                compering_point = roots[i]
                real_roots.append(roots[i])

        real_roots.sort()
        return real_roots



    def no_double(self, roots, func,maxeror):
            if len(roots) == 0:
                return []
            
            points_to_check = [roots[0]]
            compering_point = roots[0]
            
            for i in range(1, len(roots)):
                f1 = func(compering_point)
                f2 = func(roots[i])
                deltaX = abs(compering_point - roots[i])  ## |Xcompre - Xi|
                dev1 = self.dev_at_point(func,compering_point)
                dev2 = self.dev_at_point(func,roots[i])
                
                if f1 * f2 < 0:  # The Intermediate value theorem holds
                    points_to_check.append(roots[i])
                    compering_point = roots[i]
                    
                elif deltaX > maxeror and dev1 * dev2 <= 0:  # Close but with derivatives in the opposite sign
                    points_to_check.append(roots[i])
                    compering_point = roots[i]
                    
                elif abs(f2) <= maxeror:  # A point very close to zero
                    points_to_check.append(roots[i])
                    compering_point = roots[i]
                    
            return points_to_check


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
        yroots = []
        #xroots = []

        points = list(np.linspace(a, b, 2500 + round(b-a)))
        points = self.no_double(points, g, maxerr)

        for point in points:
            root = self.newton_raphson(g, point, maxerr, a, b)
            if root is not None:
                yroots.append(root)
              #  xroots.append(point)

        if len(yroots) > 0:
            roots = self.select_roots(yroots, g, maxerr)

        else:
            return []

        return list(set(roots))

