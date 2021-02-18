import simple_math


def test_simple_add():
    assert simple_math.simple_add(1,2) == 3

def test_simple_sub():
    assert simple_math.simple_sub(1,2) == -1

def test_simple_mult():
    assert simple_math.simple_mult(1,2) == 2

def test_simple_div():
    assert simple_math.simple_div(1,2) == 0.5
        
def test_poly_first():
    assert simple_math.poly_first(1,2,3) == 5

def test_poly_second():
    assert simple_math.poly_second(1,2,3,4) == 9
    assert simple_math.poly_second(2,2,3,4) == 24

