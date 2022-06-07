from turtle import st
from HNF import HNF, HNF_full_row_rank
from CVP import greedy, branch_and_bound, projection
from wolfram import get_HNF_from_wolfram_alpha
import numpy as np
from datetime import datetime
from utils import generate_random_array, generate_random_linearly_independent_float_matrix, generate_random_linearly_independent_matrix, generate_random_matrix, generate_random_matrix_with_full_row_rank, generate_random_float_matrix, print_matrix


def HNF_problem():

    # n >= m, if matrix should be full row rank
    m = 9  # size of vector
    n = 9  # number of vectors
    lowest = 0
    highest = 6

    # generate matrix with full row rank or random matrix
    # columns of generated matrix B are vectors
    # B = [b1, .., bn]
    # bi = |bi1|
    #      |bi2|
    #      |...|
    #      |bim|
    #
    # B = [b11, .., bn1]
    #     [b12, .., bn2]
    #     ..............
    #     [b1m, .., bnm]
    # B = generate_random_matrix_with_full_row_rank(m, n, lowest, highest)

    # B = np.array([[2, 1, 2], [4, 3, 3], [4, 1, 1]])

    # print("B = \n{}".format(B))
    # print("B.T = \n{}".format(B.T))

    # # get HNF of full ranked B
    # H = HNF_full_row_rank(B)
    # H_str = get_HNF_from_wolfram_alpha(B.T)

    # print("HNF(B) = \n{}\n".format(H))
    # print("HNF(B).T = \n{}\n".format(H.T))
    # print("HNF(B) from wolfram alpha = \n{}".format(H_str))

    m = 8  # size of vector
    n = 8  # number of vectors
    lowest = 0
    highest = 6

    B = generate_random_matrix(m, n, lowest, highest)
    print("B = \n{}".format(B))
    print("B.T = \n{}".format(B.T))

    H = HNF(B)
    H_str = get_HNF_from_wolfram_alpha(B.T)
    print("HNF(B) = \n{}".format(H))
    print("HNF(B).T = \n{}".format(H.T))
    print("HNF(B) from wolfram alpha = \n{}".format(H_str))


def CVP_problem():

    m = 15  # vector size
    n = 15  # number of vectors
    lowest = 1
    highest = 25

    arr_lowest = 1
    arr_highest = 25

    # generate matrix with linear independent vectors
    # vectors of generated matrix B are vectors
    # B = |b1|
    #     |b2|
    #     |..|
    #     |bn|
    #
    # bi = [bi1, bi2, .. bim]
    #
    # B = [b11, .., b1m]
    #     [b21, .., b2m]
    #     ..............
    #     [bn1, .., bnm]

    B = generate_random_linearly_independent_matrix(
        n, m, lowest, highest)
    #print("B = \n{}".format(B))

    t = generate_random_array(m, arr_lowest, arr_highest)
    #print("t = {}".format(t))

    #print("Result of greedy: {}".format(greedy(B, t)))
    start = datetime.now()
    res = greedy(B, t)
    end = datetime.now()
    print(end - start)
    start = datetime.now()
    res = branch_and_bound(B, t)
    end = datetime.now()
    print(end - start)

    #print("Result of b&b: {}".format(branch_and_bound(B, t)))

    # B = generate_random_linearly_independent_float_matrix(
    #     n, m, lowest, highest)
    # print("B = \n{}".format(B))

    # t = generate_random_array(m, arr_lowest, arr_highest)
    # print("t = {}".format(t))

    # print("Result of greedy: {}".format(greedy(B, t)))
    # print("Result of b&b: {}".format(branch_and_bound(B, t)))


def main_problem_HNF():
    m = 20  # size of vector
    n = 20  # number of vectors
    lowest = 1
    highest = 15

    B = generate_random_matrix(m, n, lowest, highest)
    #print("B = \n{}".format(B))

    start = datetime.now()
    H = HNF(B)
    end = datetime.now()

    print(end - start)
    #H_str = get_HNF_from_wolfram_alpha(B.T)
    #print("HNF(B) = \n{}".format(H))
    #print("HNF(B).T = \n{}".format(H.T))
    #print("HNF(B) from wolfram alpha = \n{}".format(H_str))


def equality_problem_HNF():
    B = np.array([[1, 0, 3], [2, 1, -1]])
    B_stroke = np.array([[1, 2, 3], [3, 2, 2]])

    H1 = HNF(B)
    H2 = HNF(B_stroke)

    print("B = \n{}".format(B))
    print("B' = \n{}".format(B_stroke))
    print("HNF(B) = \n{}".format(H1))
    print("HNF(B') = \n{}".format(H2))


def union_of_lattices_problem_HNF():
    B = np.array([[1, 0, 3], [2, 1, -1]])
    B_stroke = np.array([[1, 2, 3], [3, 2, 2]])
    arr = np.append(B, B_stroke, axis=1)

    H = HNF(arr)

    print("B = \n{}".format(B))
    print("B' = \n{}".format(B_stroke))
    print("B|B' = \n{}".format(arr))
    print("HNF(B|B') = \n{}".format(H))


def containtment_problem_HNF():
    B = np.array([[1, 0, 3], [2, 1, -1]])
    B_stroke = np.array([[1, 2, 3], [3, 2, 2]])
    arr = np.append(B, B_stroke, axis=1)

    H1 = HNF(arr)
    H2 = HNF(B)

    print("B = \n{}".format(B))
    print("B' = \n{}".format(B_stroke))
    print("B|B' = \n{}".format(arr))
    print("HNF(B') = \n{}".format(H1))
    print("HNF(B|B') = \n{}".format(H2))


def membership_problem_HNF():
    B = np.array([[1, 0, 3], [2, 1, -1]])
    vector = np.array([[1, 1]])

    arr = np.append(B, vector.T, axis=1)

    H1 = HNF(arr)
    H2 = HNF(B)

    print("B = \n{}".format(B))
    print("v = \n{}".format(vector))
    print("B|v = \n{}".format(arr))
    print("HNF(v) = \n{}".format(H1))
    print("HNF(B|v) = \n{}".format(H2))


def main():
    # main_problem_HNF()
    # equality_problem_HNF()
    # union_of_lattices_problem_HNF()
    # containtment_problem_HNF()
    # membership_problem_HNF()
    HNF_problem()
    # CVP_problem()


if __name__ == '__main__':
    main()
