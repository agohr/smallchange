from fractions import Fraction
from copy import deepcopy
from functools import reduce
from math import gcd
from random import randint

import argparse

def fraction_matrix_inverse(matrix):
    n = len(matrix)
    matrix = [[Fraction(el) for el in row] for row in matrix]  # Convert elements to Fractions
    inverse = [[Fraction(int(i==j), 1) for i in range(n)] for j in range(n)]  # Initialize inverse matrix as identity matrix

    # Augment the matrix with the identity matrix
    matrix = [row + i_row for row, i_row in zip(matrix, inverse)]

    # Gaussian elimination
    for i in range(n):
        # Find maximum in this column
        max_el = abs(matrix[i][i])
        max_row = i
        for k in range(i+1, n):
            if abs(matrix[k][i]) > max_el:
                max_el = abs(matrix[k][i])
                max_row = k

        # Swap maximum row with current row
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]

        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            c = -matrix[k][i]/matrix[i][i]
            for j in range(i, n*2):
                if i == j:
                    matrix[k][j] = 0
                else:
                    matrix[k][j] += c * matrix[i][j]

    # Solve equation Ax=b for an upper triangular matrix A
    for i in range(n-1, -1, -1):
        for k in range(i-1, -1, -1):
            c = -matrix[k][i]/matrix[i][i]
            for j in range(n*2):
                if j == i:
                    matrix[k][j] = 0
                else:
                    matrix[k][j] += c * matrix[i][j]
        matrix[i] = [el / matrix[i][i] for el in matrix[i]]

    # Separate the matrix from its augmentation
    for i in range(n):
        matrix[i] = matrix[i][n:]

    return matrix

# Test the function with a matrix
def test_matrix_inversion():
    matrix = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    inverse = [[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]
    assert fraction_matrix_inverse(matrix) == inverse
    print("Test of matrix inversion code passed")

def create_vandermonde_matrix(points):
    n = len(points)
    vandermonde_matrix = []
    for i in range(n):
        row = [Fraction(points[i][0], 1) ** j for j in range(n)]
        vandermonde_matrix.append(row)
    return vandermonde_matrix

def get_polynomial_coefficients(points):
    n = len(points)
    matrix = create_vandermonde_matrix(points)
    matrix_inverse = fraction_matrix_inverse(matrix)
    y_values = [[Fraction(points[i][1], 1)] for i in range(n)]
    coefficients = multiply_matrices(matrix_inverse, y_values)
    return [coef[0] for coef in coefficients]

def multiply_matrices(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

# Test the function for polynomial coefficient recovery
def test_polynomial_coefficients():
    points = [(1, 1), (2, 4), (3, 9)]
    assert get_polynomial_coefficients(points) == [0, 0, 1]
    print("Test of polynomial coefficient recovery passed")

# Write a function for polynomial evaluation
def evaluate_polynomial(coefficients, x):
    return sum(coef*x**i for i, coef in enumerate(coefficients))


# write a simple function that counts the number of ways to make amount of money with coins of the available denominations.
def change_possibilities(n, denominations, memo = dict()):
    if n == 0 and n%denominations[0] == 0:
        return 1
    if n < 0:
        return 0
    if len(denominations) == 0:
        return 0
    if (n, len(denominations)) in memo:
        return memo[(n, len(denominations))]
    memo[(n, len(denominations))] = change_possibilities(n - denominations[-1], denominations, memo) + change_possibilities(n, denominations[:-1], memo)
    return memo[(n, len(denominations))]

def test_change_possibilities():
    n = change_possibilities(100, [1, 2, 5, 10, 20, 50, 100])
    assert n == 4563
    print("Test of simple change counting function passed")

def lcm(a,b):
    return abs(a*b) // gcd(a,b)

def lcm_list(l):
    return reduce(lcm, l)

def change_polynomial(n, denominations):
    lcm_denominations = lcm_list(denominations)
    n_reduced = n % lcm_denominations
    d = len(denominations)
    support_points = [(i * lcm_denominations + n_reduced, change_possibilities(i * lcm_denominations + n_reduced, denominations)) for i in range(d)]
    poly = get_polynomial_coefficients(support_points)
    return poly

# print the polynomials in standard form
# e.g. [0,1,2] should become 2x^2 + x
def print_polynomial(poly):
    poly = [str(coef) + "x^" + str(i) for i, coef in enumerate(poly) if coef != 0]
    poly = " + ".join(poly)
    poly = poly.replace("x^0", "")
    poly = poly.replace("x^1", "x")
    poly = poly.replace("1x", "x")
    poly = poly.replace(" + -", " - ")
    print(poly)

def change_number_rec(n, denominations, memo_points = dict(), memo_poly = dict(), verbose=False):
    lcm_denominations = lcm_list(denominations)
    n_reduced = n % lcm_denominations
    d = len(denominations)
    max_x_support = (d-1) * lcm_denominations + n_reduced
    if n == 0 and n%denominations[0] == 0:
        return 1
    if n < 0:
        return 0
    if len(denominations) == 0:
        return 0
    if (n, d) in memo_points:
        return memo_points[(n, d)]
    if (n_reduced, d) in memo_poly:
        f = memo_poly[(n_reduced, d)]
        res = evaluate_polynomial(f, n)
        memo_points[(n, d)] = res
        if verbose:
            print(f"C({n}, {denominations}) = {res} by polynomial evaluation")
            print("Polynomial:")
            print_polynomial(f)
            print(f"Polynomial valid for: x = {n_reduced} + {lcm_denominations} * k, k = 0, 1, 2, ...")
        return res
    if max_x_support < n:
        support_points = [(i * lcm_denominations + n_reduced, change_number_rec(i * lcm_denominations + n_reduced, denominations, memo_points, memo_poly, verbose)) for i in range(d)]
        poly = get_polynomial_coefficients(support_points)
        memo_poly[(n_reduced, d)] = poly
        res = evaluate_polynomial(poly, n)
        memo_points[(n, d)] = res
        if verbose:
            print(f"C({n}, {denominations}) = {res} by polynomial evaluation")
            print("Polynomial:")
            print_polynomial(poly)
            print(f"Polynomial valid for: x = {n_reduced} + {lcm_denominations} * k, k = 0, 1, 2, ...")
        return res
    else:
        res1 = change_number_rec(n, denominations[:-1], memo_points, memo_poly, verbose)
        res2 = change_number_rec(n - denominations[-1], denominations, memo_points, memo_poly, verbose)
        res = res1 + res2
        memo_points[(n, d)] = res
        if verbose:
            print(f"C({n}, {denominations}) = {res} by recursive evaluation")
        return res

def test_change_polynomial():
    r = randint(1, 10000)
    poly = change_polynomial(r, [1, 2, 5, 10, 20, 50, 100])
    T = evaluate_polynomial(poly, r)
    T1 = change_possibilities(r, [1, 2, 5, 10, 20, 50, 100])
    assert T == T1
    print("Test of polynomial construction passed")


def test_change_number_rec():
    r = randint(1, 500)
    T = change_number_rec(r, [1, 2, 5, 10, 20, 50, 100], verbose=True)
    T1 = change_possibilities(r, [1, 2, 5, 10, 20, 50, 100])
    assert T == T1
    print("Test of recursive change number computation passed")

def test_all():
    test_matrix_inversion()
    test_polynomial_coefficients()
    test_change_possibilities()
    test_change_polynomial()
    test_change_number_rec()

if __name__ == '__main__':
    test_all()
    
