import numpy as np
import numpy.ma as ma
from utils import max_plus, min_plus
import time
import copy
from collections import Counter

np.random.seed(0)

class FastSTMF:
    """
    Fit a fast sparse tropical matrix factorization model for a matrix X.
    such that
        A = U V + E
    where
        A is of shape (m, n)    - data matrix
        U is of shape (m, rank) - approximated row space
        V is of shape (rank, n) - approximated column space
        E is of shape (m, n)    - residual (error) matrix
    """

    def __init__(self, rank=5, initialization='random_vcol',
                 epsilon=0.00000000000001, random_acol_param=5, threshold=300, seed_param=42):
        """
        :param rank: Rank of the matrices of the model.
        :param max_iter: Maximum nuber of iterations.
        """
        self.rank = rank
        self.initialization = initialization
        self.epsilon = epsilon
        self.random_acol_param = random_acol_param
        self.threshold = threshold
        self.is_transposed = False
        self.seed_param = seed_param

    def b_norm(self, A):
        return np.sum(np.abs(A))

    def initialize_U(self, A, m):
        U_initial = np.zeros((m, self.rank))
        k = self.random_acol_param  # number of columns to average
        if self.initialization == 'random':
            low = A.min()
            high = A.max()
            U_initial = low + (high - low) * np.random.rand(m, self.rank)  # uniform distribution
        elif self.initialization == 'random_vcol':
            # each column in U is element-wise average(mean) of a random subset of columns in A
            for s in range(self.rank):
                U_initial[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].mean(axis=1)
        elif self.initialization == 'col_min':
            for s in range(self.rank):
                U_initial[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].min(axis=1)
        elif self.initialization == 'col_max':
            for s in range(self.rank):
                U_initial[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].max(axis=1)
        elif self.initialization == 'scaled':
            low = np.min(A)  # c
            high = 0
            U_initial = low + (high - low) * np.random.rand(m, self.rank)
        return ma.masked_array(U_initial, mask=np.zeros((m, self.rank)))

    def assign_values(self, U, V, f, iterations, rows_perm, time, uvu, vuv, ulf, urf, errors, times):
        rows_perm_inverse = np.argsort(rows_perm)
        if self.is_transposed == True:
            self.U, self.V = V.T, U[rows_perm_inverse,:].T
        else:
            self.U, self.V = U[rows_perm_inverse,:], V
        self.error = f
        self.iterations = iterations
        self.time = time
        self.count_uvu, self.count_vuv = uvu, vuv
        self.count_ulf, self.count_urf = ulf, urf
        self.errors, self.times = errors, times

    def ULF(self, U, A, V, ind_j, ind_k, i):
        # UVU
        U_old_k_vector = copy.deepcopy(U[:, ind_k])  # copying k-th column from U
        V_old_k_vector = copy.deepcopy(V[ind_k, :])  # copying k-th row from V

        U[i, ind_k] = A[i, ind_j] - V[ind_k, ind_j] # inplace change

        mat_U = np.negative(U[:, ind_k]).reshape((1, -1))
        V[ind_k, :] = min_plus(mat_U, A).reshape((-1))
        mat_V = np.negative(V[ind_k, :]).reshape((-1, 1))
        U[:, ind_k] = min_plus(A, mat_V).reshape((-1))

        f_new = self.b_norm(np.subtract(A, max_plus(U, V)))
        return U, V, f_new, U_old_k_vector, V_old_k_vector

    def URF(self, U, A, V, ind_j, ind_k, i):
        # VUV
        U_old_k_vector = copy.deepcopy(U[:, ind_k])  # copying k-th column from U
        V_old_k_vector = copy.deepcopy(V[ind_k, :])  # copying k-th row from V

        V[ind_k, ind_j] = A[i, ind_j] - U[i, ind_k]  # inplace change
        mat_V = np.negative(V[ind_k, :]).reshape((-1, 1))
        U[:, ind_k] = min_plus(A, mat_V).reshape((-1))
        mat_U = np.negative(U[:, ind_k]).reshape((1, -1))
        V[ind_k, :] = min_plus(mat_U, A).reshape((-1))
        f_new = self.b_norm(np.subtract(A, max_plus(U, V)))
        return U, V, f_new, U_old_k_vector, V_old_k_vector

    def return_old_U_and_V(self, U, V, U_old_k_vector, V_old_k_vector, ind_k):
        U[:, ind_k] = U_old_k_vector
        V[ind_k, :] = V_old_k_vector
        return

    def compute_td_element(self, i, j, A, U, V):
        mask, k_list, trop_k = A.mask, range(self.rank), []
        for k in k_list:
            trop_k.append((U[i, k] + V[k, j]))
        max_k, max_k_elem_index = max(trop_k), np.argmax(trop_k)
        td_element = np.abs(A[i, j] - max_k)
        return td_element, max_k_elem_index

    def compute_td_row(self, i, A, U, V):
        m, n = A.shape  # mxn
        mask, td_row, j_list, td_row_k_indices = A.mask, 0, range(n), []
        for j in j_list:
            if not mask[i, j]:
                td_element, max_k_elem_index = self.compute_td_element(i, j, A, U, V)
                td_row += td_element
                td_row_k_indices.append((j, max_k_elem_index))
        return td_row, td_row_k_indices

    def compute_td_column(self, j, A, U, V):
        m, n = A.shape  # mxn
        mask, td_column, i_list, td_column_k_indices = A.mask, 0, range(m), []
        for i in i_list:
            if not mask[i, j]:
                td_element, max_k_elem_index = self.compute_td_element(i, j, A, U, V)
                td_column += td_element
                td_column_k_indices.append((j, max_k_elem_index))
        return td_column, td_column_k_indices

    def count_k(self, td_row_k_indices, td_column_k_indices):
        c_row = Counter([e[1] for e in td_row_k_indices])
        c_col = Counter([e[1] for e in td_column_k_indices])
        c_sum = c_row + c_col
        k_common = c_sum.most_common(1).pop()
        return k_common[0], k_common[1]

    def fit(self, A):
        """
        Fit model parameters U, V.
        :param A:
            Data matrix of shape (m, n)
            Unknown values are assumed to be masked.
        """
        basic_time = time.time()
        start_time = time.time()
        # check if matrix is wide
        m, n = A.shape  # rows, columns

        if m > n: # tall matrix
            self.is_transposed = True
            A = A.T # wide
            m, n = A.shape

        # permute matrix A, random rows
        rows_perm = np.random.RandomState(seed=self.seed_param).permutation(m)
        A = A[rows_perm, :]

        iterations = 0
        uvu, vuv = 0, 0
        ulf, urf = 0, 0
        errors, times = [], []

        # initialization of U matrix
        U_initial = self.initialize_U(A, m)
        V = min_plus(ma.transpose(np.negative(U_initial)), A)
        U = min_plus(A, ma.transpose(np.negative(V)))
        D = np.subtract(A, max_plus(U, V))

        # initialization of f values needed for convergence test
        norm = self.b_norm(D)
        f_old = norm + self.epsilon + 1
        f_new = norm
        f = f_new
        # save initial error
        current_time = time.time() - start_time
        errors.append(f)
        times.append(round(current_time, 3))
        start_time = time.time()

        i_list, j_list, k_list = range(m), range(n), range(self.rank)
        mask = A.mask

        while (f_old - f_new) > self.epsilon:
            f = f_new
            iterations += 1
            for i in i_list:
                td_row, td_row_k_indices = self.compute_td_row(i, A, U, V)
                err, err_indices = [], []
                for j in j_list:
                    if not mask[i, j]:
                        td_column, td_column_k_indices = self.compute_td_column(j, A, U, V)
                        err.append(td_column)
                        err_indices.append(td_column_k_indices)

                n_err = len(err)
                diffs_indices = sorted(range(n_err), key=lambda x: err[x], reverse=True)

                for index in range(n_err):  # finding the element which decreases error
                    temp = diffs_indices[index]
                    temp_j = err_indices[temp][0]
                    ind_j = temp_j[0]
                    ind_k, _ = self.count_k(td_row_k_indices, err_indices[temp])

                    U, V, f_new, U_old_k_vector, V_old_k_vector = self.ULF(U, A, V, ind_j, ind_k, i)
                    ulf += 1
                    # save error and time for 1 iteration
                    current_time = time.time() - start_time
                    errors.append(f_new if f_new < f else f)
                    times.append(round(current_time, 3))
                    start_time = time.time()

                    if f_new < f:
                        uvu += 1
                        break
                    self.return_old_U_and_V(U, V, U_old_k_vector, V_old_k_vector, ind_k)
                    if (time.time() - basic_time) >= self.threshold:
                        self.assign_values(U, V, f, iterations, rows_perm, round(time.time() - start_time, 3), uvu, vuv, ulf, urf, errors, times)
                        # save error and time for last iteration
                        current_time = time.time() - start_time
                        errors.append(f_new if f_new < f else f)
                        times.append(round(current_time, 3))
                        start_time = time.time()
                        return

                    U, V, f_new, U_old_k_vector, V_old_k_vector = self.URF(U, A, V, ind_j, ind_k, i)
                    urf += 1
                    # save error and time for 1 iteration
                    current_time = time.time() - start_time
                    errors.append(f_new if f_new < f else f)
                    times.append(round(current_time, 3))
                    start_time = time.time()

                    if f_new < f:
                        vuv += 1
                        break
                    self.return_old_U_and_V(U, V, U_old_k_vector, V_old_k_vector, ind_k)
                    if (time.time() - basic_time) >= self.threshold:
                        self.assign_values(U, V, f, iterations, rows_perm, round(time.time() - start_time, 3), uvu, vuv, ulf, urf, errors, times)
                        # save error and time for last iteration
                        current_time = time.time() - start_time
                        errors.append(f_new if f_new < f else f)
                        times.append(round(current_time, 3))
                        start_time = time.time()
                        return

                if f_new < f:
                    f_old, f = f, f_new
                    if (time.time() - basic_time) >= self.threshold:
                        self.assign_values(U, V, f, iterations, rows_perm, round(time.time() - start_time, 3), uvu, vuv, ulf, urf, errors, times)
                        # save error and time for last iteration
                        current_time = time.time() - start_time
                        errors.append(f_new if f_new < f else f)
                        times.append(round(current_time, 3))
                        start_time = time.time()
                        return
                else:
                    print("no solution found!")
                    self.assign_values(U, V, f, iterations, rows_perm, round(time.time() - start_time, 3), uvu, vuv, ulf, urf, errors, times)
                    return

        print("Fast STMF achieved the convergence by epsilon.")
        self.assign_values(U, V, f, iterations, rows_perm, round(time.time() - start_time, 3), uvu, vuv, ulf, urf, errors, times)


    def predict_all(self):
        """
        Return approximated matrix for all
        columns and rows.
        """
        return max_plus(self.U, self.V)

    def get_statistics(self, version, s, j, folder, transpose=False):
        results = [self.iterations, self.count_uvu, self.count_vuv, self.count_ulf, self.count_urf, self.time]
        if transpose == False:
            np.savetxt(folder + version + "/" + str(s) + "_" + str(j) + ".csv", results)
            np.savetxt(folder + version + "/errors/" + str(s) + "_" + str(j) + "_errors.csv", self.errors)
            np.savetxt(folder + version + "/times/" + str(s) + "_" + str(j) + "_times.csv", self.times)
        else:
            np.savetxt(folder + version + "/" + str(s)  + "_" + str(j) + "_transpose.csv", results)
            np.savetxt(folder + version + "/errors/" + str(s) + "_" + str(j) + "_errors_transpose.csv", self.errors)
            np.savetxt(folder + version + "/times/" + str(s) + "_" + str(j) + "_times_transpose.csv", self.times)
        return
