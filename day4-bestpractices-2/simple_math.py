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
    """ 
    Returns the product of two numbers. 
    

    Parameters
    ----------
    a : Multiplier.
    b : Multiplicand.

    See Also
    --------
    simple_div
    
    """
    return a*b

def simple_div(a,b):
    """ 
    Returns the quotient of two numbers. 
    
    Parameters
    ----------
    a : Nominator.
    b : Denominator.
    
    See Also
    --------
    simple_mult
    
    """
    return a/b

def poly_first(x, a0, a1):
    """ 
    Returns the value of a first order polynomial with parameters
    a0 and a1 evaluated at x. 
    
    Parameters
    ----------
    x : Variable value for the evaluation of the function.
    a0 : Constant term.
    a1 : Coefficient multiplying the linear term.
   
    See Also
    --------
    poly_second
    
    Examples
    --------
    >>> simple_math.poly_first(1,2,3)
    5

    """
    return a0 + a1*x

def poly_second(x, a0, a1, a2):
    """ 
    Returns the value of a second order polynomial with parameters
    a0, a1, and a2 evaluated at x. 
    
    Parameters
    ----------
    x : Variable value for the evaluation of the function.
    a0 : Constant term.
    a1 : Coefficient multiplying the linear term.
    a2 : Coefficient multiplying the quadratic term.
    
    See Also
    --------
    poly_first

    Examples
    --------
    >>> simple_math.poly_second(1,2,3,4)
    9
   

    """
    return poly_first(x, a0, a1) + a2*(x**2)

# Feel free to expand this list with more interesting mathematical operations...
# .....
