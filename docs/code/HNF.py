import numpy as np
from numpy import linalg
from utils import *


def HNF_full_row_rank(matrix):  # Hermite Normal Form of full rank matrix

    m = matrix.shape[0]
    n = matrix.shape[1]

    # inds - indexes of linear independent columns
    # number of such rows is equal to m, so B_stroke always m x m
    temp, inds = sympy.Matrix(matrix).rref()

    B_stroke = []
    for i in inds:
        B_stroke.append(matrix.T[i])

    B_stroke = (np.array(B_stroke)).T

    det = round(det_by_gram_schmidt(B_stroke))

    H = np.eye(m, dtype=int) * det

    for i in range(n):
        H = add_column(H, matrix.T[i])

    num_of_additional_zero_vectors = n - m

    zero_vector = np.array([np.zeros(m, dtype=int)])

    for i in range(num_of_additional_zero_vectors):
        H = np.append(H.T, zero_vector, axis=0).T

    return H


def add_column(H, b_column):
    if (H.shape[1] == 0):
        return H

    a = H[0, 0]
    h = H[1:, 0]
    H_stroke = H[1:, 1:]
    b = b_column[0]
    b_stroke = b_column[1:]

    g, x, y = gcd_extended(a, b)

    U = np.array([[x, -b / g], [y, a / g]])
    first_column = np.append(np.array(a), h)
    second_column = np.append(np.array(b), b_stroke)
    temp_matrix = np.array([first_column, second_column]).T
    temp_result = np.dot(temp_matrix, U)

    h_stroke = temp_result.T[0, 1:]
    b_double_stroke = temp_result.T[1, 1:]

    b_double_stroke = reduce(b_double_stroke, H_stroke)

    H_double_stroke = add_column(H_stroke, b_double_stroke)

    h_stroke = reduce(h_stroke, H_double_stroke)

    length_of_zeros = H_double_stroke.shape[0]

    result = np.array(np.append(np.array([np.append(np.array(g), np.zeros(length_of_zeros, dtype=int))]), np.append(
        np.array([h_stroke]), H_double_stroke.T, axis=0).T, axis=0), dtype=int)

    return result


def HNF(matrix):
    m = matrix.shape[0]
    n = matrix.shape[1]

    B_stroke = projection(matrix)

    B_double_stroke = HNF_full_row_rank(B_stroke)

    result = inverse_projection(B_double_stroke, matrix)

    return result
