import sys
import numpy as np
import mysymnmf as symn

np.random.seed(1234)

def parse_arguments():
    """
    Parses the command line arguments.
    :return: Tuple containing the number of clusters (k), goal (operation), and file name
    """
    k = None 
    goal = None
    file_name = None
    k = sys.argv[1]
    file_name = sys.argv[3]
    goal = sys.argv[2]
    return k, goal, file_name


def read_array_from_file(filename):
    """
    Reads a 2D array of points from the given file.
    :param filename: The name of the file to read from
    :return: 2D list of floats representing the data
    """
    with open(filename, 'r') as file:
        lines = file.readlines()        
        array = [
            [float(value) for value in line.strip().split(',')]
            for line in lines
        ]
    return array


def main():
    """
    Main function to execute the program logic based on what goal is chosen.
    it reads input data, processes it according to the specified goal,
    and prints the result.
    """
    try:
        k, goal, file_name = parse_arguments()
        pnt_array = read_array_from_file(file_name)
        rows = len(pnt_array)
        cols = len(pnt_array[0])
        if goal == "symnmf":
            norm_matrix = symn.norm(pnt_array)
            total_sum = sum(sum(row) for row in norm_matrix)
            num_elements = sum(len(row) for row in norm_matrix)
            average = total_sum / num_elements
            upper_bound = 2 * np.sqrt(average / int(float(k)))
            H = np.random.uniform(0, upper_bound, (rows, int(float(k)))).tolist()
            final_H = symn.symnmf(H, norm_matrix, int(float(k)), rows)
            for row in final_H:
                print(','.join('%.4f' % value for value in row))
        elif goal == "sym":
            sym_matrix = symn.sym(pnt_array)
            for row in sym_matrix:
                print(','.join('%.4f' % value for value in row))
        elif goal == "ddg":
            diag_matrix = symn.ddg(pnt_array)
            for row in diag_matrix:
                print(','.join('%.4f' % value for value in row))
        elif goal == "norm":
            norm_matrix = symn.norm(pnt_array)
            for row in norm_matrix:
                print(','.join('%.4f' % value for value in row))
    except Exception as e:
        print("An Error Has Occurred")
        exit()

if __name__ == "__main__":
    main()
