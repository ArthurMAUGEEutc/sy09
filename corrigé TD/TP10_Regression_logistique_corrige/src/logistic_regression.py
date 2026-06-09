import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils import check_X_y

# La fonction logistique
from scipy.special import expit


# On hérite de `LinearClassifierMixin` qui gère toute la partie
# prédiction pourvu qu'on renseigne les attributs `coef_`,
# `intercept_` et `classes_` lors de l'apprentissage.
class LogisticRegression(BaseEstimator, LinearClassifierMixin):
    def __init__(self, max_iter=1000, tol=1e-10, fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # On valide `X` et `y` en transformant par exemple les
        # DataFrame ou Series Pandas en tableau Numpy
        X, y = check_X_y(X, y)

        # On définit les classes qui ne sont pas nécessairement les
        # entiers 0 et 1. On convertit les classes de `y` en des
        # entiers.
        self.classes_, y = np.unique(y, return_inverse=True)

        # On rajoute une colonne de "1" si on veut apprendre une
        # constante
        p = X.shape[1]
        if self.fit_intercept:
            p += 1
            X = np.column_stack((np.ones(X.shape[0]), X))

        it = 1
        grad = self.tol + 1
        beta = np.zeros(p)

        while np.max(np.abs(grad)) > self.tol and it < self.max_iter:
            # Calcul des p_i avec la fonction logistique
            # <answer>
            pi = expit(X @ beta)
            # </answer>

            # Calcul du gradient
            # <answer>
            grad = X.T @ (y - pi)
            # </answer>

            # Calcul de la matrice hessienne
            # <answer>
            W = np.diag(pi * (1 - pi))
            hessian = -X.T @ W @ X
            # </answer>

            # Calcul de l'incrément avec `np.linalg.solve` ou
            # `np.linalg.inv`.
            # <answer>
            step = -np.linalg.solve(hessian, grad)
            # </answer>

            # Mise à jour de beta
            # <answer>
            beta += step
            # </answer>

            it += 1

        # Stockage des paramètres appris. Attention, pour que la
        # prédiction avec la fonction `predict` marche, il faut que
        # l'attribut `coef_` soit une matrice ligne. De même,
        # l'attribut `intercept_` même si c'est un scalaire doit être
        # mis dans un tableau. Il faut également tenir compte de
        # l'option `fit_intercept`.
        # <answer>
        self.coef_ = beta[None, 1:] if self.fit_intercept else beta[None, :]
        self.intercept_ = beta[[0]] if self.fit_intercept else np.array([0])
        # </answer>

        # On retourne `self` pour pouvoir éventuellement chainer avec un `predict`.
        return self
