#ifndef LINKER_H_
#define LINKER_H_

/**
 * Creates a 2D matrix of size n*k initialized to zero.
 * @param n: Number of rows
 * @param k: Number of columns
 * @return: Pointer to the allocated matrix
 */
double** create_matrix(int n, int k);

/**
 * Frees the allocated memory for a 2D matrix.
 * @param matrix: Pointer to the matrix to free
 * @param n: Number of rows in the matrix
 */
void free_matrix(double** matrix, int n);

/**
 * Multiplies two matrices.
 * @param M1: First matrix
 * @param M2: Second matrix
 * @param n: Number of rows in M1
 * @param m: Number of columns in M1 and rows in M2
 * @param q: Number of columns in M2
 * @return: Pointer to the resulting matrix
 */
double** mat_mult(double** M1, double** M2, int n,int m, int q); 

/**
 * Calculates the squared Euclidean distance between two points.
 * @param point1: First point
 * @param point2: Second point
 * @param dim: Dimensionality of the points
 * @return: Squared Euclidean distance
 */
double squared_euc_dis(double *point1, double *point2, int dim);

/**
 * Computes the similarity matrix from a set of points.
 * 
 * @param points: Pointer to the array of points
 * @param n: Number of points
 * @param dim: Dimensionality of each point
 * @return: Pointer to the similarity matrix
 */
double** sym_mat(double**points,int n, int dim);

/**
 * Computes the diagonal degree matrix from a similarity matrix.
 * 
 * @param A: Similarity matrix
 * @param n: Size of the matrix
 * @return: Pointer to the diagonal degree matrix
 */
double** diag_mat(double**A,int n);

/**
 * Normalizes a similarity matrix using the diagonal degree matrix.
 * @param D: Diagonal degree matrix
 * @param A: Similarity matrix
 * @param n: Size of the matrices
 * @return: Pointer to the normalized matrix
 */
double** norm_mat(double**D,double**A, int n);

/**
 * Computes the average of the entries in a matrix.
 * @param M: Pointer to the matrix
 * @param n: Size of the matrix
 * @return: Average value of the matrix entries
 */
double mat_entry_avg(double** M,int n);

/**
 * Transposes a matrix.
 * @param M: Pointer to the matrix
 * @param n: Number of rows
 * @param m: Number of columns
 * @return: Pointer to the transposed matrix
 */
double** transpose_matrix(double** M, int n,int m);

/**
 * Calculates the Frobenius norm of a matrix.
 * @param M: Pointer to the matrix
 * @param n: Number of rows
 * @param m: Number of columns
 * @return: Frobenius norm
 */
double forb(double** M,int n,int m);

/**
 * performnig the calculations needed to compute H when calling opt_mat_with_H using the matrices H and W.
 * @param H: Matrix H
 * @param W: Matrix W
 * @param n: Number of rows in H
 * @param k: Number of columns in H
 * @param mone: Matrix mone
 * @param mechane: Matrix mechane
 * @param old_H: Matrix old_H
 * @return: an integer =0 if memory allocation failed, else returns integer = 1
 */
double calcul(double **H, double** W, int n, int k,double**mone,double** mechane,double** old_H);

/**
 * Optimizes the matrix H using the matrices H and W.
 * @param H: Matrix H
 * @param W: Matrix W
 * @param n: Number of rows in H
 * @param k: Number of columns in H
 * @return: Pointer to the optimized matrix H
 */
double** opt_mat_with_H(double **H, double**W,int n,int k);

/**
 * Prints a matrix to the console.
 * @param res: Pointer to the matrix
 * @param rows: Number of rows
 * @param cols: Number of columns
 */
void printMatrix(double** res, int rows, int cols);

/**
 * Counts the number of rows in a file.
 * @param filename: Name of the file
 * @return: Number of rows in the file
 */
int count_rows_from_file(char* filename);

/**
 * Counts the number of columns in a file.
 * @param filename: Name of the file
 * @return: Number of columns in the first row of the file
 */
int count_cols_from_file(char* filename);

/**
 * Creates a 2D array from a file.
 * @param filename: Name of the file
 * @param rows: Number of rows
 * @param cols: Number of columns
 * @return: Pointer to the created array
 */
double** create_array_from_file(char* filename, int rows, int cols);

#endif
