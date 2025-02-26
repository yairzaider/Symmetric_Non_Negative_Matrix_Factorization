#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define EPSILON 0.0001
#define MAXITER 300

/**
 * Creates a 2D matrix of size n*k initialized to zero.
 * @param n: Number of rows
 * @param k: Number of columns
 * @return: Pointer to the allocated matrix
 */
double** create_matrix(int n, int k) 
{
    int i;
    int j;
    double** matrix = (double**)calloc(n, sizeof(double*));
    if (matrix == NULL) 
    {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    for (i = 0; i < n; i++) 
    {
        matrix[i] = (double*)calloc(k, sizeof(double));
        if (matrix[i] == NULL) 
        {
            printf("An Error Has Occurred\n");
            for (j = 0; j < i; j++) 
            {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix; 
}

/**
 * Frees the allocated memory for a 2D matrix.
 * @param matrix: Pointer to the matrix to free
 * @param n: Number of rows in the matrix
 */
void free_matrix(double** matrix, int n) 
{
    int i;
    if (matrix != NULL) 
    {
        for (i = 0; i < n; i++) 
        {
            free(matrix[i]);
        }
        free(matrix);
    }
}

/**
 * Multiplies two matrices.
 * @param M1: First matrix
 * @param M2: Second matrix
 * @param n: Number of rows in M1
 * @param m: Number of columns in M1 and rows in M2
 * @param q: Number of columns in M2
 * @return: Pointer to the resulting matrix
 */
double** mat_mult(double** M1, double** M2, int n, int m, int q) 
{
    int i;
    int j;
    int t;
    double** M3 = create_matrix(n, q);
    if (M3 == NULL){
        free_matrix(M1, n);
        free_matrix(M2, m);
        return NULL;
    }
    for (i = 0; i < n; i++){
        for (j = 0; j < q; j++) 
        {
            M3[i][j] = 0.0;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < q; j++) 
        {
            for (t = 0; t < m; t++) 
            {
                M3[i][j] += M1[i][t] * M2[t][j];
            }
        }
    }
    return M3; 
}

/**
 * Calculates the squared Euclidean distance between two points.
 * @param point1: First point
 * @param point2: Second point
 * @param dim: Dimensionality of the points
 * @return: Squared Euclidean distance
 */
double squared_euc_dis(double *point1, double *point2, int dim)
{
    double accu = 0.0;
    int counter = 0;
    while (counter < dim)
    {
        accu += pow((point1[counter] - point2[counter]), 2);
        counter++;
    }
    return accu;
}

/**
 * Computes the similarity matrix from a set of points.
 * 
 * @param points: Pointer to the array of points
 * @param n: Number of points
 * @param dim: Dimensionality of each point
 * @return: Pointer to the similarity matrix
 */
double** sym_mat(double** points, int n, int dim)
{
    int i;
    int j;
    double** A = create_matrix(n, n);
    if (A == NULL)
    {
        free_matrix(points, n);
        return NULL;
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i != j)
            {
                A[i][j] = exp((-0.5) * squared_euc_dis(points[i], points[j], dim));
            }
            else
            {
                A[i][j] = 0.0;
            }
        }
    }
    return A;
}

/**
 * Computes the diagonal degree matrix from a similarity matrix.
 * 
 * @param A: Similarity matrix
 * @param n: Size of the matrix
 * @return: Pointer to the diagonal degree matrix
 */
double** diag_mat(double** A, int n)
{
    double row_sum = 0.0;
    int i;
    int j;
    double** D = create_matrix(n, n);
    if (D == NULL)
    {
        free_matrix(A, n);
        return NULL;
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            row_sum += A[i][j];
        }
        D[i][i] = row_sum;
        row_sum = 0.0;
    }
    return D;
}

/**
 * Normalizes a similarity matrix using the diagonal degree matrix.
 * @param D: Diagonal degree matrix
 * @param A: Similarity matrix
 * @param n: Size of the matrices
 * @return: Pointer to the normalized matrix
 */
double** norm_mat(double** D, double** A, int n)
{
    int i;
    double** rev_sqr_D = create_matrix(n, n);
    double** temp_res;
    double** res;
    if (rev_sqr_D == NULL){
        free_matrix(D, n);
        free_matrix(A, n);
        return NULL;
    }
    for (i = 0; i < n; i++){
        rev_sqr_D[i][i] = 1 / (sqrt(D[i][i]));
    }
    temp_res = mat_mult(rev_sqr_D, A, n, n, n);
    if (temp_res == NULL){
        free_matrix(D, n);
        return NULL;
    }
    res = mat_mult(temp_res, rev_sqr_D, n, n, n);
    if (res == NULL){
        free_matrix(D, n);
        free_matrix(A, n);
        return NULL;
    }
    free_matrix(temp_res, n);
    free_matrix(rev_sqr_D, n);
    return res;
}

/**
 * Computes the average of the entries in a matrix.
 * @param M: Pointer to the matrix
 * @param n: Size of the matrix
 * @return: Average value of the matrix entries
 */
double mat_entry_avg(double** M, int n)
{
    int i;
    int j;
    double sum = 0.0;
    double avg = 0.0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            sum += M[i][j];
        }
    }
    avg = sum / (n * n);
    return avg;
}

/**
 * Transposes a matrix.
 * @param M: Pointer to the matrix
 * @param n: Number of rows
 * @param m: Number of columns
 * @return: Pointer to the transposed matrix
 */
double** transpose_matrix(double** M, int n, int m) 
{
    int i;
    int j;
    double** trans = create_matrix(m, n);
    if (trans == NULL)
    {
        free_matrix(M, n);
        return NULL;
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            trans[j][i] = M[i][j];
        }
    }
    return trans;
}

/**
 * Calculates the Frobenius norm of a matrix.
 * @param M: Pointer to the matrix
 * @param n: Number of rows
 * @param m: Number of columns
 * @return: Frobenius norm
 */
double forb(double** M, int n, int m)
{
    int i;
    double** temp_mat;
    double trace = 0.0;
    double** trans_mat = transpose_matrix(M, n, m);
    temp_mat = mat_mult(trans_mat, M, m, n, m);
    for (i = 0; i < m; i++)
    {
        trace += temp_mat[i][i];
    }
    return trace;
}

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
double calcul(double **H, double** W, int n, int k,double**mone,double** mechane,double** old_H)
{
    int m;
    int i;
    int j;
    for (m = 0; m < MAXITER; m++) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                old_H[i][j] = H[i][j];}}
        mone = mat_mult(W, old_H, n, n, k);
        if (mone == NULL) {
            if (H != NULL) { free_matrix(H, n); }
            if (mechane != NULL) { free_matrix(mechane, n); }
            return 0;}
        mechane = mat_mult(mat_mult(old_H, transpose_matrix(old_H, n, k), n, k, n), old_H, n, n, k);
        if (mechane == NULL) {
            free_matrix(H, n);
            free_matrix(mone, n);
            return 0; }
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H[i][j] = old_H[i][j] * (0.5 + 0.5 * (mone[i][j] / mechane[i][j]));}}
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                old_H[i][j] = H[i][j] - old_H[i][j];}}
        if (forb(old_H, n, k) < EPSILON) { break; }}
        return 1;
}

/**
 * Optimizes the matrix H using the matrices H and W.
 * @param H: Matrix H
 * @param W: Matrix W
 * @param n: Number of rows in H
 * @param k: Number of columns in H
 * @return: Pointer to the optimized matrix H
 */
double ** opt_mat_with_H(double **H, double** W, int n, int k)
{
    double** mone = NULL;
    double** mechane = NULL;
    double** old_H = NULL;
    old_H = create_matrix(n, k);
    if (old_H == NULL) 
    {
        free_matrix(H, n);
        free_matrix(W, n);
        return NULL;}
    if(calcul(H,  W, n,  k, mone, mechane, old_H) ==0)
    {
        return NULL;
    }
    else
    {
        free_matrix(old_H, n);
        free_matrix(mone, n);
        free_matrix(mechane, n);
        return H;
    }
}
 
/**
 * Prints a matrix to the console.
 * @param res: Pointer to the matrix
 * @param rows: Number of rows
 * @param cols: Number of columns
 */
void printMatrix(double** res, int rows, int cols) 
{
    int i;
    int j;
    for (i = 0; i < rows; i++) 
    {
        for (j = 0; j < cols; j++) 
        {
            if (j == cols - 1) 
            {
                printf("%.4f", res[i][j]); 
            } else 
            {
                printf("%.4f,", res[i][j]); 
            }
        }
        printf("\n"); 
    }
}

/**
 * Counts the number of rows in a file.
 * @param filename: Name of the file
 * @return: Number of rows in the file
 */
int count_rows_from_file(char* filename)
{
    char ch;
    int rows = 0;
    FILE *file;
    file = fopen(filename, "r");
    if (!file)
    {
        return -1;
    }
    while (fscanf(file, "%c", &ch) == 1)
    {
        if (ch == '\n')
        {
            rows += 1;
        }
    }
    fclose(file);
    return rows;
}

/**
 * Counts the number of columns in a file.
 * @param filename: Name of the file
 * @return: Number of columns in the first row of the file
 */
int count_cols_from_file(char* filename)
{
    char ch;
    int cols = 1;
    FILE *file;
    file = fopen(filename, "r");
    if (!file)
    {
        return -1;
    }
    while (fscanf(file, "%c", &ch) == 1 && ch != '\n') 
    {
        if (ch == ',') 
        {
            cols++;
        }
    }
    fclose(file);
    return cols;
}

/**
 * Creates a 2D array from a file.
 * @param filename: Name of the file
 * @param rows: Number of rows
 * @param cols: Number of columns
 * @return: Pointer to the created array
 */
double** create_array_from_file(char* filename, int rows, int cols)
{
    int i;
    int j;
    double** final_array = create_matrix(rows, cols);
    FILE *file;
    file = fopen(filename, "r");
    if (!file){
        printf("An Error Has Occurred\n");
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++){
            if (j == cols - 1) {
                if (fscanf(file, "%lf", &final_array[i][j]) != 1) {
                    fclose(file);
                    free_matrix(final_array, rows);
                    return NULL;  
                }
            } else 
            {
                if (fscanf(file, "%lf,", &final_array[i][j]) != 1) {
                    fclose(file);
                    free_matrix(final_array, rows);
                    return NULL;  
                }
            }
        }
    }
    fclose(file);  
    return final_array;  
}


/**
 * Main function for the program, checks if reading the inputs went ok and calls the right functions according to the chosen goal.
 * @param argc: Argument count
 * @param argv: Argument vector
 * @return: Exit status, 0 if ok, 1 if error
 */
int main(int argc, char* argv[])
{   double** pnt_arr;
    int rows;
    int cols;
    char* goal = argv[1];
    double** tmp_mat1;
    double** tmp_mat2;
    double** res;
    if (argc < 3) {
        printf("An Error Has Occurred\n");
        return 1;}
    rows = count_rows_from_file(argv[2]);
    cols = count_cols_from_file(argv[2]);
    if (rows == -1 || cols == -1){
        printf("An Error Has Occurred\n");
        return 1;}
    pnt_arr = create_array_from_file(argv[2], rows, cols);
    if(pnt_arr==NULL){return 1;}
    if (strcmp(goal, "sym") == 0){res = sym_mat(pnt_arr, rows, cols);}
    else{ if (strcmp(goal, "ddg") == 0){
            tmp_mat1 = sym_mat(pnt_arr, rows, cols);
            res = diag_mat(tmp_mat1, rows);
            free_matrix(tmp_mat1, rows);} 
        else{ tmp_mat1 = sym_mat(pnt_arr, rows, cols);
            tmp_mat2 = diag_mat(tmp_mat1, rows);
            res = norm_mat(tmp_mat2, tmp_mat1, rows);
            free_matrix(tmp_mat1, rows);
            free_matrix(tmp_mat2, rows);}}
    printMatrix(res, rows, rows);
    free_matrix(res, rows);
    free_matrix(pnt_arr, rows);
    return 0;
}
