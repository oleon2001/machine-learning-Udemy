matrix = [[1,2,3],
          [4,5,6],
          [7,8,9]]


transposed = [[row[x] for row in matrix] for x in range(len(matrix[0]))]


print(transposed)