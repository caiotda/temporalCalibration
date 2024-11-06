import math
import numpy as np

from utils import load, save


class SVDpp:
    def __init__(self, train_set, test_set, output_path, n_factors, lr=0.05, reg=0.02, n_epochs=10):
        self.train_set = load(train_set)
        self.test_set = load(test_set)
        self.n_factors = n_factors
        self.output_path = output_path
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.global_mean = train_set['rating'].mean()
        self.n_users = len(train_set['userId'].unique())
        self.n_items = len(train_set['movieId'].unique())
        self.bu = np.zeros(self.n_users)  # User biases
        self.bi = np.zeros(self.n_items)  # Item biases
        self.p = np.random.normal(0.1, 0.1, (self.n_users, n_factors))  # Users factor matrix
        self.q = np.random.normal(0.1, 0.1, (self.n_items, n_factors))  # Items factor matrix
        self.implicit_factor = np.random.normal(0.1, 0.1, (self.n_users, n_factors))  # Implicit feedback factors

    def _interacted(self, u):
        """
        Given a userId u, returns a dataframe containing
        only the movies the user has interacted with.
        """
        return self.train_set[self.train_set['userId'] == u]

    def eta(self, u):
        """
        Given a userId u, calculates the amount of 
        items the user has interacted with
        """
        return len(self._interacted(u))

    def interacted_movies_by(self, u):
        """
        Given a userId u, returns a list of unique
        movieIds the user has interacted with
        """
        return self._interacted(u)['movieId'].unique()

    def get_implicit_term(self, u):
        """
        Given a userId u, returns the implicit feedback related
        term for the svd++ prediction. For this, we combine
        the item factors, stored in p, that the user has
        interacted.
        """
        eta_u = self.eta(u)
        movies_interacted = self.interacted_movies_by(u)
        implicit_sum = 0
        for movieId in movies_interacted:
            implicit_sum += self.p[movieId]
        return implicit_sum / math.sqrt(eta_u), eta_u

    def predict(self, u, i):
        modified_user_factor = self.p[u] + self.implicit_factor[u]
        predict = self.global_mean + self.bu[u] + self.bi[i] + np.dot(modified_user_factor, self.q[i])
        return predict

    def train(self):
        error = []
        for t in range(self.n_epochs):
            sq_error = 0
            j = 0
            print(f"Starting iter {t + 1}/{self.n_epochs}")
            for index, row in self.train_set.iterrows():
                if j % 1000 == 0:
                    print(f"Iteration {j} of {len(self.train_set)}: ({100 * (j / len(self.train_set)):.2f}%)")
                u = row['userId']
                i = row['movieId']
                implicit_term_u, eta_u = self.get_implicit_term(u)
                modified_user_factor = self.p[u] + implicit_term_u

                pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(modified_user_factor, self.q[i])

                r_ui = row['rating']
                e_ui = r_ui - pred
                sq_error += e_ui ** 2

                # Update params
                self.bu[u] = self.bu[u] + self.lr * e_ui - self.reg * self.bu[u]
                self.bi[i] = self.bi[i] + self.lr * e_ui - self.reg * self.bi[i]
                j += 1
                for f in range(self.n_factors):
                    temp_uf = self.p[u][f]

                    self.implicit_factor[u][f] = self.implicit_factor[u][f] + self.lr * (
                            (e_ui * self.q[i][f] / math.sqrt(eta_u)) - self.reg * self.implicit_factor[u][f])
                    self.p[u][f] = self.p[u][f] + self.lr * (e_ui * self.q[i][f] - self.reg * self.p[u][f])
                    self.q[i][f] = self.q[i][f] + self.lr * (e_ui * temp_uf - self.reg * self.q[i][f])
            error.append(math.sqrt(sq_error / len(self.train_set)))
        print("Training finished")
        self.model_params = {"mu": self.global_mean, "bu": self.bu, "bi": self.bi, "p": self.p, "q": self.q, "imp_y": self.implicit_factor}
        return self.model_params, error