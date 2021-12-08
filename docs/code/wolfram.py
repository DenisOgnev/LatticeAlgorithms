import wolframalpha


def get_HNF_from_wolfram_alpha(matrix):

    app_id = '55HY54-H7V5VVRTGY'

    client = wolframalpha.Client(app_id)

    query = "HermiteDecomposition[{{"

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (j == matrix.shape[1] - 1):
                query += str(matrix[i, j])
            else:
                query += str(matrix[i, j]) + ", "
        if (i == matrix.shape[0] - 1):
            query += "}}]"
        else:
            query += "}, {"

    res = client.query(query)

    answer = next(res.results).text
    answer = str(answer)

    index = answer.find("H =")

    result = answer[index:]

    result = result.replace("H = ", "")
    result = result.replace("(", "")
    result = result.replace(")", "")

    return result
