import math
from utils import *


def projection(matrix, vector):
    projection = np.zeros(vector.shape)

    for i in range(0, matrix.shape[0]):
        projection += (scalar(vector, matrix[i]) /
                       scalar(matrix[i], matrix[i])) * matrix[i]
    result = vector - projection

    return result


def closest_vector(matrix, vector):
    closest = matrix[0]
    for v in matrix:
        if (distance_between_two_vectors(vector, v) < distance_between_two_vectors(vector, closest)):
            closest = v

    return closest


def greedy(matrix, target):
    if (matrix.shape[0] == 0):
        return 0

    b = matrix[-1]
    matrix = matrix[:-1]
    b_star = projection(matrix, b)
    x = scalar(target, b_star) / scalar(b_star, b_star)
    c = round(x)

    return c * b + greedy(matrix, target - c * b)


def branch_and_bound(matrix, target):
    if (matrix.shape[0] == 0):
        return 0
    b = matrix[-1]
    matrix = matrix[:-1]
    v = greedy(matrix, target)
    b_star = projection(matrix, b)

    #x_array = np.array([2, 3])
    x_array = []
    v_array = []

    upper_bound = math.ceil(norm(target - v))
    x_middle = math.floor(scalar(target, b_star) / scalar(b_star, b_star))
    lower_bound = norm(projection(matrix, target - x_middle * b))

    x = x_middle
    temp_lower_bound = lower_bound
    while (temp_lower_bound <= upper_bound):
        x += 1
        temp_lower_bound = norm(projection(matrix, target - x * b))
    x_highest = x

    x = x_middle
    temp_lower_bound = lower_bound
    while (temp_lower_bound <= upper_bound):
        x -= 1
        temp_lower_bound = norm(projection(matrix, target - x * b))
    x_lowest = x + 1

    for i in range(x_lowest, x_highest):
        x_array.append(i)

    if (len(x_array) == 0):
        x_array.append(x_middle)

    for x in x_array:
        res = x * b + branch_and_bound(matrix, target - x * b)
        v_array.append(res)
    v_array = np.array(v_array)

    return closest_vector(v_array, target)
