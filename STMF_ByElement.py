import numpy as np
import numpy.ma as ma
from utils import max_plus, min_plus, get_coordinates
import time
import copy

np.random.seed(0)

class STMF:
    """
    Fit a sparse tropical matrix factorization model for a matrix X.
    such that
        A = U V + E
    where
        A is of shape (m, n)    - data matrix
        U is of shape (m, rank) - approximated row space
        V is of shape (rank, n) - approximated column space
        E is of shape (m, n)    - residual (error) matrix
    """

    def __init__(self, rank=5, criterion='convergence', max_iter=100, initialization='random_vcol',
                 epsilon=0.00000000000001, random_acol_param=5, threshold=300):
        """
        :param rank: Rank of the matrices of the model.
        :param max_iter: Maximum nuber of iterations.
        """
        self.rank = rank
        self.max_iter = max_iter
        self.initialization = initialization
        self.epsilon = epsilon
        self.random_acol_param = random_acol_param
        self.criterion = criterion # convergence or iterations
        self.threshold = threshold
        self.is_transposed = False

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

    def assign_values(self, U, V, f, iterations, columns_perm, time, uvu, vuv, ulf, urf, errors, times):
        columns_perm_inverse = np.argsort(columns_perm)
        if self.is_transposed == True:
            self.U, self.V = V[:, columns_perm_inverse].T, U.T
        else:
            self.U, self.V = U, V[:, columns_perm_inverse]
        self.error = f
        self.iterations = iterations
        self.time = time
        self.count_uvu, self.count_vuv = uvu, vuv
        self.count_ulf, self.count_urf = ulf, urf
        self.errors, self.times = errors, times
        
    def ULF(self, U, A, V, ind_k, i, ind_j):
        # UVU
        U_old_k_vector = copy.deepcopy(U[:, ind_k])  # copying k-th column from U
        V_old_k_vector = copy.deepcopy(V[ind_k, :])  # copying k-th row from V

        U[i, ind_k] = A[i, ind_j] - V[ind_k, ind_j]  # inplace change

        mat_U = np.negative(U[:, ind_k]).reshape((1, -1))
        V[ind_k, :] = min_plus(mat_U, A).reshape((-1))
        mat_V = np.negative(V[ind_k, :]).reshape((-1, 1))
        U[:, ind_k] = min_plus(A, mat_V).reshape((-1))

        f_new = self.b_norm(np.subtract(A, max_plus(U, V)))
        #print(f_new)
        return U, V, f_new, U_old_k_vector, V_old_k_vector

    def URF(self, U, A, V, ind_k, i, ind_j):
        # VUV
        U_old_k_vector = copy.deepcopy(U[:, ind_k])  # copying k-th column from U
        V_old_k_vector = copy.deepcopy(V[ind_k, :])  # copying k-th row from V

        V[ind_k, ind_j] = A[i, ind_j] - U[i, ind_k]  # inplace change
        mat_V = np.negative(V[ind_k, :]).reshape((-1, 1))
        U[:, ind_k] = min_plus(A, mat_V).reshape((-1))
        mat_U = np.negative(U[:, ind_k]).reshape((1, -1))
        V[ind_k, :] = min_plus(mat_U, A).reshape((-1))
        f_new = self.b_norm(np.subtract(A, max_plus(U, V)))
        #print(f_new)
        return U, V, f_new, U_old_k_vector, V_old_k_vector
    
    def return_old_U_and_V(self, U, V, U_old_k_vector, V_old_k_vector, ind_k):
        #print("returning old values")
        U[:, ind_k] = U_old_k_vector
        V[ind_k, :] = V_old_k_vector
        return

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
        m, n = A.shape
        
        # comment for reproducing synthetic training experiments
        if m > n: # tall matrix
            self.is_transposed = True
            A = A.T # wide
            m, n = A.shape

        # permute matrix A, columns minimum increasing
        columns_perm = np.argsort(np.min(A, axis = 0))
        A = A[:, columns_perm]

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

        k_list = range(self.rank)
        U_new, V_new = U, V
        comb = get_coordinates(A)  # known values

        while (f_old - f_new) > self.epsilon:
            f = f_new
            for x in comb:  # (i, j)
                i, j = x
                trop_dist, trop_dist_indices, trop_k = [], [], []
                for k in k_list:
                    trop_k.append((U[i, k] + V[k, j]))
                max_k, max_k_index = max(trop_k), np.argmax(trop_k)
                trop_dist.append(np.abs(A[i, j] - max_k))
                ind = max_k_index  # k index
                if trop_dist[0] == 0:  # perfect approximation, skip updating
                    continue
                U_new, V_new, f_new, U_old_k_vector, V_old_k_vector = self.ULF(U, A, V, ind, i, j)
                ulf += 1
                # save error and time for 1 iteration
                current_time = time.time() - start_time
                errors.append(f_new if f_new < f else f)
                times.append(round(current_time, 3))
                start_time = time.time()
                if f_new < f:
                    uvu += 1
                    break
                self.return_old_U_and_V(U, V, U_old_k_vector, V_old_k_vector, ind)
                if (time.time() - basic_time) >= self.threshold:
                    self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - basic_time, 3), uvu, vuv, ulf, urf, errors, times)
                    # save error and time for last iteration
                    current_time = time.time() - start_time
                    errors.append(f_new if f_new < f else f)
                    times.append(round(current_time, 3))
                    start_time = time.time()
                    return

                # VUV
                U_new, V_new, f_new, U_old_k_vector, V_old_k_vector = self.URF(U, A, V, ind, i, j)
                urf += 1
                # save error and time for 1 iteration
                current_time = time.time() - start_time
                errors.append(f_new if f_new < f else f)
                times.append(round(current_time, 3))
                start_time = time.time()
                if f_new < f:
                    vuv += 1
                    break
                self.return_old_U_and_V(U, V, U_old_k_vector, V_old_k_vector, ind)
                if (time.time() - basic_time) >= self.threshold:
                    self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - basic_time, 3), uvu, vuv, ulf, urf, errors, times)
                    # save error and time for last iteration
                    current_time = time.time() - start_time
                    errors.append(f_new if f_new < f else f)
                    times.append(round(current_time, 3))
                    start_time = time.time()
                    return

            if f_new < f:
                U, V = U_new, V_new
                f_old, f = f, f_new
                if (time.time() - basic_time) >= self.threshold:
                    self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - basic_time, 3), uvu, vuv, ulf, urf, errors, times)
                    # save error and time for last iteration
                    current_time = time.time() - start_time
                    errors.append(f_new if f_new < f else f)
                    times.append(round(current_time, 3))
                    start_time = time.time()
                    return
            else:
                print("no solution found!")
                self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - basic_time, 3), uvu, vuv, ulf, urf, errors, times)
                return
        self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - basic_time, 3), uvu, vuv, ulf, urf, errors, times)


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
            np.savetxt(folder + version + "/" + str(s) + "_" + str(j) + "_transpose.csv", results)
            np.savetxt(folder + version + "/errors/" + str(s) + "_" + str(j) + "_errors_transpose.csv", self.errors)
            np.savetxt(folder + version + "/times/" + str(s) + "_" + str(j) + "_times_transpose.csv", self.times)
        return