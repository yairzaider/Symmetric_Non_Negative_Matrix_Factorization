#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include "symnmf.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/**
 * Converts a Python list of lists (2D array) to a C double pointer (matrix like).
 * @param lst_py: Python list to convert
 * @param rows: Number of rows in the matrix
 * @param cols: Number of columns in the matrix
 * @return: Pointer to the C matrix (double**)
 */
static double** lst_Py_to_lst_c(PyObject* lst_py, int rows, int cols) {
    int i, j;
    PyObject *place_holder_lst;
    PyObject *place_holder_cord;
    double **lst_c = create_matrix(rows, cols);
    for (i = 0; i < rows; i++) {   
        place_holder_lst = PyList_GetItem(lst_py, i);
        for (j = 0; j < cols; j++) {
            place_holder_cord = PyList_GetItem(place_holder_lst, j);
            lst_c[i][j] = PyFloat_AsDouble(place_holder_cord);
        }
    }
    return lst_c;
}

/**
 * Converts a C double pointer (matrix) to a Python list of lists.
 * @param lst_c: C matrix to convert
 * @param rows: Number of rows in the matrix
 * @param cols: Number of columns in the matrix
 * @return: Python list containing the matrix
 */
static PyObject* lst_c_to_lst_Py(double** lst_c, int rows, int cols) {
    int i, j;
    PyObject* py_lst = PyList_New(rows);
    for (i = 0; i < rows; i++) {
        PyObject *temp_pnt = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyObject *temp_cord = PyFloat_FromDouble(lst_c[i][j]);
            if (PyList_SetItem(temp_pnt, j, temp_cord) < 0) {
                return NULL;
            }
        }
        if (PyList_SetItem(py_lst, i, temp_pnt) < 0) {
            return NULL;
        }
    }
    return py_lst; 
}

/**
 * Does the optimization of the matrix H using the non-negative matrix factorization.
 * @param self: Pointer to the module
 * @param args: Arguments passed from Python
 * @return: Optimized matrix H as a Python object
 */
static PyObject* opt_mat_py(PyObject *self, PyObject *args) {
    int k, rows;
    PyObject *H_py, *W_py, *final_H_py;
    double** H_c;
    double** W_c;
    double** final_H_c;
    if (!PyArg_ParseTuple(args, "OOii", &H_py, &W_py, &k, &rows)) {
        return NULL;
    }
    H_c = lst_Py_to_lst_c(H_py, rows, k);
    if (H_c == NULL) {
        return NULL;
    }
    W_c = lst_Py_to_lst_c(W_py, rows, rows);
    if (W_c == NULL) {
        free_matrix(H_c, rows);
        return NULL;
    }
    final_H_c = opt_mat_with_H(H_c, W_c, rows, k);
    final_H_py = lst_c_to_lst_Py(final_H_c, rows, k);
    free_matrix(H_c, rows);
    free_matrix(W_c, rows);
    return final_H_py; 
}

/**
 * Constructs a symmetric matrix from the input points.
 * @param self: Pointer to the module
 * @param args: Arguments passed from Python
 * @return: Symmetric matrix as a Python object
 */
static PyObject* sym_mat_py(PyObject *self, PyObject *args) {
    PyObject *pnt_lst_py;
    double **pnt_lst;
    int rows, cols;
    PyObject *final_similarity_matrix;
    if (!PyArg_ParseTuple(args, "O", &pnt_lst_py)) {
        return NULL;
    }
    rows = PyList_Size(pnt_lst_py);
    cols = PyList_Size(PyList_GetItem(pnt_lst_py, 0));
    pnt_lst = lst_Py_to_lst_c(pnt_lst_py, rows, cols);
    double** sym_mat_c = sym_mat(pnt_lst, rows, cols);
    final_similarity_matrix = lst_c_to_lst_Py(sym_mat_c, rows, rows);
    free_matrix(pnt_lst, rows);
    free_matrix(sym_mat_c, rows);
    return final_similarity_matrix; 
}

/**
 * Builds a diagonal degree matrix from the input points.
 * @param self: Pointer to the module
 * @param args: Arguments passed from Python
 * @return: Diagonal degree matrix as a Python object
 */
static PyObject* diag_mat_py(PyObject *self, PyObject *args) {
    PyObject *pnt_lst_py;
    double **pnt_lst;
    int rows, cols;
    PyObject *final_diag_deg_matrix;
    if (!PyArg_ParseTuple(args, "O", &pnt_lst_py)) {
        return NULL;
    }
    rows = PyList_Size(pnt_lst_py);
    cols = PyList_Size(PyList_GetItem(pnt_lst_py, 0));
    pnt_lst = lst_Py_to_lst_c(pnt_lst_py, rows, cols);
    double** sym_mat_c = sym_mat(pnt_lst, rows, cols);
    double** diag_mat_c = diag_mat(sym_mat_c, rows);
    final_diag_deg_matrix = lst_c_to_lst_Py(diag_mat_c, rows, rows);
    free_matrix(pnt_lst, rows);
    free_matrix(sym_mat_c, rows);
    free_matrix(diag_mat_c, rows);
    return final_diag_deg_matrix; 
}

/**
 * Normalizes a matrix using the diagonal matrix.
 * @param self: Pointer to the module
 * @param args: Arguments passed from Python
 * @return: Normalized similarity matrix as a Python object
 */
static PyObject* norm_mat_py(PyObject *self, PyObject *args) {
    PyObject *pnt_lst_py;
    double **pnt_lst;
    int rows, cols;
    PyObject *final_norm_similarity_matrix;
    if (!PyArg_ParseTuple(args, "O", &pnt_lst_py)) {
        return NULL;
    }
    rows = PyList_Size(pnt_lst_py);
    cols = PyList_Size(PyList_GetItem(pnt_lst_py, 0));
    pnt_lst = lst_Py_to_lst_c(pnt_lst_py, rows, cols);
    double** sym_mat_c = sym_mat(pnt_lst, rows, cols);
    double** diag_mat_c = diag_mat(sym_mat_c, rows);
    double** norm_mat_c = norm_mat(diag_mat_c, sym_mat_c, rows);
    final_norm_similarity_matrix = lst_c_to_lst_Py(norm_mat_c, rows, rows);
    free_matrix(pnt_lst, rows);
    free_matrix(sym_mat_c, rows);
    free_matrix(diag_mat_c, rows);
    free_matrix(norm_mat_c, rows);
    return final_norm_similarity_matrix; 
}

/**
 * Method definitions for the module.
 */
static PyMethodDef symnmf_methods[] = {
    {"symnmf", (PyCFunction)opt_mat_py, METH_VARARGS, PyDoc_STR("Optimize the matrix H")},
    {"sym", (PyCFunction)sym_mat_py, METH_VARARGS, PyDoc_STR("Construct a symmetric matrix")},
    {"ddg", (PyCFunction)diag_mat_py, METH_VARARGS, PyDoc_STR("Construct a diagonal degree matrix")},
    {"norm", (PyCFunction)norm_mat_py, METH_VARARGS, PyDoc_STR("Normalize a matrix")},
    {NULL, NULL, 0, NULL}
};

/**
 * Module definition structure.
 */
static struct PyModuleDef symmmodule = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf",
    NULL,
    -1,
    symnmf_methods
};

/**
 * Module initialization function.
 */
PyMODINIT_FUNC PyInit_mysymnmf(void) {
    PyObject *m;
    m = PyModule_Create(&symmmodule);
    if (!m) {
        return NULL;
    }
    return m;
}
