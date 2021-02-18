"""
A collection of simple math operations
"""

def simple_add(a,b):
    """ Returns the sum of two numbers."""
    return a+b

def simple_sub(a,b):
    """ Returns the difference between two numbers."""
    return a-b

def simple_mult(a,b):
    """ Returns the product of two numbers. """
    return a*b

def simple_div(a,b):
    """ Returns the quotient of two numbers. """
    return a/b

def poly_first(x, a0, a1):
    """ Returns the value of a first order polynomial with parameters
    a0 and a1 evaluated at x. """
    return a0 + a1*x

def poly_second(x, a0, a1, a2):
    """ Returns the value of a second order polynomial with parameters
    a0, a1, and a2 evaluated at x. """
    return poly_first(x, a0, a1) + a2*(x**2)

# Feel free to expand this list with more interesting mathematical operations...
# .....
