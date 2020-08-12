def l2_norm(sample):
    if not isinstance(sample, list): return abs(sample)
    sq_sample = [ x ** 2 for x in sample ]
    return pow(sum(sq_sample), 0.5)

def abs_error(y, y_hat):
    if not isinstance(y, list): y, y_hat = [y], [y_hat]
    return [ x - x_hat for x, x_hat in zip(y, y_hat) ]

def relative_error(y, y_hat):
    error = abs_error(y, y_hat)
    l2 = l2_norm(error)
    return l2 / l2_norm(y_hat)

def length(A, B):
    return l2_norm(abs_error(A, B))

def test_relative_error():
    # Points are equal
    # test_case_re([0,0], [0,0], 0.0) division by zero
    test_case_re([1,1], [1,1], 0.0)
    test_case_re([10, 15], [10,15], 0.0)
    test_case_re([-10, -15], [-10, -15], 0.0)
    test_case_re([10, -15], [10,-15], 0.0)
    test_case_re([-10, 15], [-10, 15] , 0.0)
    # Pointwise
    test_case_re(1.0, 2.0, 0.50)
    test_case_re(2.0, 1.0, 1.00)
    test_case_re(-1.0, 2.0, 3. / 2.)
    test_case_re(1.0, -2.0, 3. / 2.)
    # Normal use cases 2D
    test_case_re([1.0, 1.0], [2.0, 2.0], 0.50)
    test_case_re([2.0, 2.0], [1.0, 1.0], 1.00)
    test_case_re([79.0, 74.0], [61.0, 83.0], 0.195)
    test_case_re([79.0, -74.0], [61.0, -83.0], 0.195)
    test_case_re([-79.0, 74.0], [-61.0, 83.0], 0.195)

def test_length_calculation():
    # Points are equal
    # test_case_length([0,0], [0,0], 0.0) division by zero
    test_case_length([1,1], [1,1], 0.0)
    test_case_length([10, 15], [10,15], 0.0)
    test_case_length([-10, -15], [-10, -15], 0.0)
    test_case_length([10, -15], [10,-15], 0.0)
    test_case_length([-10, 15], [-10, 15] , 0.0)
    # Pointwise
    test_case_length(1.0, 2.0, 1.00)
    test_case_length(2.0, 1.0, 1.00)
    test_case_length(-1.0, 2.0, 3.00)
    test_case_length(1.0, -2.0, 3.00)
    # Normal use cases 2D
    test_case_length([1.0, 1.0], [2.0, 2.0], 2.00 ** 0.5)
    test_case_length([2.0, 2.0], [1.0, 1.0], 2.00 ** 0.5)
    test_case_length([79.0, 74.0], [61.0, 83.0], 20.125)
    test_case_length([79.0, -74.0], [61.0, -83.0], 20.125)
    test_case_length([-79.0, 74.0], [-61.0, 83.0], 20.125)

def test_case_re(y, y_hat, expected, epsilon=0.01):
    assert(relative_error(y, y_hat) < expected + epsilon)
    assert(relative_error(y, y_hat) > expected - epsilon)

def test_case_length(a, b, expected, epsilon=0.01):
    assert(length(a, b) < expected + epsilon)
    assert(length(a, b) > expected - epsilon)

if __name__ == '__main__':
    test_relative_error()
    test_length_calculation()