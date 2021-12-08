import numpy as np
import sympy
from sympy.matrices import matrices
from sympy.matrices.dense import Matrix


def scalar(vector1, vector2):
    s = 0
    for i in range(vector1.size):
        s += vector1[i] * vector2[i]
    return s


def norm(vector):
    return np.sqrt(scalar(vector, vector))


def distance_between_two_vectors(vector1, vector2):
    return np.sqrt(scalar(vector1 - vector2, vector1 - vector2))


def generate_random_linearly_independent_matrix(n, m, lowest, highest):
    if (n > m):
        raise("Number of vectors should be less or equal than their size")
    matrix = np.random.randint(lowest, highest, (n, m))

    while not check_linear_independent(matrix):
        matrix = np.random.randint(lowest, highest, (n, m))

    return matrix


def generate_random_linearly_independent_float_matrix(n, m, lowest, highest):
    if (n > m):
        raise("Number of vectors should be less or equal than their size")
    matrix = np.random.uniform(lowest, highest, (n, m))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = round(matrix[i, j], 1)

    while not check_linear_independent(matrix):
        matrix = np.random.uniform(lowest, highest, (n, m))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = round(matrix[i, j], 1)

    return matrix


def generate_random_array(m, lowest, highest):
    array = np.random.uniform(lowest, highest, m)

    for i in range(array.shape[0]):
        array[i] = round(array[i], 1)

    return array


# matrix is full row rank if rank(matrix) == m
def generate_random_matrix_with_full_row_rank(m, n, lowest, highest):
    matrix = np.random.randint(lowest, highest, (m, n))
    rank = get_matrix_rank(matrix)

    while m != rank:
        matrix = np.random.randint(lowest, highest, (m, n))
        rank = get_matrix_rank(matrix)

    return matrix


def generate_random_matrix(m, n, lowest, highest):
    matrix = np.random.randint(lowest, highest, (m, n))

    return matrix


def generate_random_float_matrix(m, n, lowest, highest):
    matrix = np.random.uniform(lowest, highest, (m, n))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = round(matrix[i, j], 2)

    return matrix


def get_matrix_rank(matrix):
    return np.linalg.matrix_rank(matrix)


def check_linear_independent(matrix):
    # inds - indexes of linear independent rows
    temp, inds = sympy.Matrix(matrix.T).rref()
    number_of_linear_independent_rows = len(inds)
    number_of_vectors = matrix.shape[0]

    if (number_of_linear_independent_rows != number_of_vectors):
        return False

    return True


def gram_schmidt(matrix, normalize=False, delete_zero_rows=True):
    basis = []
    for vector in matrix:
        projections = np.sum(
            (scalar(vector, b) / scalar(b, b)) * b for b in basis)
        r = vector - projections
        is_all_zero = np.all(r == 0)
        if (delete_zero_rows):
            if (not is_all_zero):
                if (normalize):
                    basis.append(r / scalar(r, r))
                else:
                    basis.append(r)
        else:
            if (normalize):
                basis.append(r / scalar(r, r))
            else:
                basis.append(r)
    return np.array(basis)


def det_by_gram_schmidt(matrix):
    result = 1
    matrix = gram_schmidt(matrix.T)
    for v in matrix:
        result *= norm(v)

    return result


def gcd_extended(a, b):
    if a == 0:
        return b, 0, 1

    gcd, x1, y1 = gcd_extended(b % a, a)

    x = y1 - (b//a) * x1
    y = x1

    return gcd, x, y


def sign(a):
    if np.sign(a) == 1 or np.sign(a) == 0:
        return True
    else:
        return False


def reduce(vector, matrix):
    for i in range(vector.shape[0]):
        while(vector[i] < 0):
            vector += matrix.T[i]
        while(vector[i] >= matrix[i, i]):
            vector -= matrix.T[i]

    return vector


def projection(matrix):
    # inds - indexes of linear independent rows
    temp, inds = sympy.Matrix(matrix.T).rref()

    result = []
    for i in inds:
        result.append(matrix[i])

    result = np.array(result)

    return result


def inverse_projection(HNF, matrix):
    # inds - indexes of linear independent rows
    temp, inds = sympy.Matrix(matrix.T).rref()

    basis = HNF

    independent_vectors = []

    for i in inds:
        independent_vectors.append(matrix[i])

    independent_vectors = np.array(independent_vectors)

    for i in range(matrix.shape[0]):
        if i not in inds:
            x = np.linalg.solve(independent_vectors.T, matrix[i])

            result = np.zeros_like(x)

            for j in range(HNF.shape[0]):
                result += HNF[j] * x[j]

            result = np.array([result])

            basis = np.append(basis, result, axis=0)

    return basis


def print_matrix(matrix):
    res = str()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (j == matrix.shape[1] - 1):
                if (i != matrix.shape[0] - 1):
                    res += str(matrix[i, j]) + "\n"
                else:
                    res += str(matrix[i, j])
            else:
                res += str(matrix[i, j]) + " | "
    print(res)
