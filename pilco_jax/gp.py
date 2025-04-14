import jax.numpy as jnp
from jax import value_and_grad
from jax.numpy.linalg import solve, cholesky, det
from scipy.optimize import minimize

from pilco_jax import Empty
from pilco_jax.util import maha


class Kernel:
    def __init__(self):
        pass

    def __add__(self, other):
        sum = Kernel()
        sum.sub = self, other
        sum.num_hyp = lambda x: self.num_hyp(x) + other.num_hyp(x)
        return sum

    def __call__(self, loghyp, x, z=None):
        loghyp = jnp.atleast_2d(loghyp)
        left, right = self.sub
        L = left.num_hyp(x)
        return left(loghyp[:, :L], x, z) + right(loghyp[:, L:], x, z)


class Kernel_RBF(Kernel):
    def __init__(self):
        super().__init__()
        self.num_hyp = lambda x: jnp.size(x, 1) + 1

    def __call__(self, loghyp, x, z=None):
        loghyp = jnp.atleast_2d(loghyp)
        n, D = x.shape
        ell = jnp.exp(loghyp[:, :D])
        sf2 = jnp.exp(2 * loghyp[:, D]).reshape(-1, 1, 1)

        x_ell = jnp.expand_dims(x, 0) / jnp.expand_dims(ell, 1)
        if z is None:
            diff = jnp.expand_dims(x_ell, 1) - jnp.expand_dims(x_ell, 2)
        else:
            z_ell = jnp.expand_dims(z, 0) / jnp.expand_dims(ell, 1)
            diff = jnp.expand_dims(x_ell, 1) - jnp.expand_dims(z_ell, 2)

        K = sf2 * jnp.exp(-0.5 * jnp.sum(diff**2, axis=3))
        return K


class Kernel_C(Kernel):
    def __init__(self):
        super().__init__()
        self.num_hyp = lambda x: 1

    def __call__(self, loghyp, x, z=None):
        loghyp = jnp.atleast_2d(loghyp)
        n, _ = x.shape
        s2 = jnp.exp(2 * loghyp).reshape(-1, 1, 1)
        if z is None:
            K = s2 * jnp.expand_dims(jnp.eye(n), 0)
        else:
            K = 0
        return K


class GPModel:
    def __init__(self, kernel=None):
        self.kernel = kernel if kernel else Kernel_RBF() + Kernel_C()

    def gp0(self, m, s):
        return self.gp2(m, s)

    
    def log_pdf(self, hyp):
        x = jnp.atleast_2d(self.inputs)
        y = jnp.atleast_2d(self.targets)

        n, D = x.shape
        _, E = y.shape

        hyp = hyp.reshape(E, -1)
        K = self.kernel(hyp, x)
        L = cholesky(K)
        alpha = jnp.stack([solve(K[i], y[:, i]) for i in range(E)], axis=1)
        y_flat = y.flatten(order='F')

        logp = 0.5 * n * E * jnp.log(2 * jnp.pi) + 0.5 * jnp.dot(y_flat, alpha.flatten()) \
               + jnp.sum(jnp.log(jnp.diagonal(L, axis1=1, axis2=2)))
        return logp

    def hyp_crub(self, hyp):
        x = jnp.atleast_2d(self.inputs)
        y = jnp.atleast_2d(self.targets)

        n, D = x.shape
        _, E = y.shape
        hyp = hyp.reshape(E, -1)
        p = 30

        ll = hyp[:, :D]
        lsf = hyp[:, D:-1]
        lsn = hyp[:, -1]

        L = self.log_pdf(hyp)
        L += jnp.sum(((ll - jnp.log(self.curb.std)) / jnp.log(self.curb.ls))**p)
        L += jnp.sum(((lsf - lsn) / jnp.log(self.curb.snr))**p)
        return L

    def cache(self):
        x = jnp.atleast_2d(self.inputs)
        y = jnp.atleast_2d(self.targets)
        n, E = y.shape

        self.K = self.kernel(self.hyp, x)
        self.iK = jnp.stack([solve(self.K[i], jnp.eye(n)) for i in range(E)])
        self.alpha = jnp.stack([solve(self.K[i], y[:, i]) for i in range(E)], axis=1)

    def optimize(self, curb=None):
        x = jnp.atleast_2d(self.inputs)
        y = jnp.atleast_2d(self.targets)

        n, D = x.shape
        _, E = y.shape

        if curb:
            self.curb = curb
        elif not hasattr(self, "curb"):
            self.curb = Empty()
            self.curb.snr = 500
            self.curb.ls = 100
            self.curb.std = jnp.std(x, axis=0)

        if not hasattr(self, "hyp"):
            self.hyp = jnp.zeros((E, D + 2))
            self.hyp = self.hyp.at[:, :D].set(jnp.log(jnp.std(x, axis=0)))
            self.hyp = self.hyp.at[:, D].set(jnp.log(jnp.std(y, axis=0)))
            self.hyp = self.hyp.at[:, -1].set(jnp.log(jnp.std(y, axis=0) / 10))

        print("Train hyperparameters of full GP...")
        hyp_flat = self.hyp.flatten()

        result = minimize(value_and_grad(self.hyp_crub), hyp_flat, jac=True)
        self.result = result
        self.hyp = result.x.reshape(E, -1)
        self.cache()

    def gp2(self, m, s):
        self.cache()
        x = jnp.atleast_2d(self.inputs)
        y = jnp.atleast_2d(self.targets)
        n, D = x.shape
        _, E = y.shape

        X = self.hyp
        beta = self.alpha

        m = jnp.atleast_2d(m)
        inp = x - m

        iL = jnp.stack([jnp.diag(jnp.exp(-X[i, :D])) for i in range(E)])
        iN = inp @ iL
        B = iL @ s @ iL + jnp.eye(D)
        t = jnp.stack([solve(B[i].T, iN[i].T).T for i in range(E)])
        q = jnp.exp(-0.5 * jnp.sum(iN * t, axis=2))
        qb = q * beta.T
        tiL = t @ iL
        c = jnp.exp(2 * X[:, D]) / jnp.sqrt(det(B))

        M = jnp.sum(qb, axis=1) * c
        V = (jnp.transpose(tiL, [0, 2, 1]) @ qb[..., None]).reshape(E, D).T * c
        k = 2 * X[:, D].reshape(E, 1) - jnp.sum(iN**2, axis=2) / 2

        inp = inp[None, ...] / jnp.exp(2 * X[:, :D])[:, None, :]
        ii = jnp.repeat(inp[:, None, :, :], E, 1)
        ij = jnp.repeat(inp[None, :, :, :], E, 0)

        iL = jnp.stack([jnp.diag(jnp.exp(-2 * X[i, :D])) for i in range(E)])
        siL = iL[None, :, :, :] + iL[:, None, :, :]
        R = s @ siL + jnp.eye(D)
        t = 1 / jnp.sqrt(det(R))
        iRs = jnp.stack([solve(R.reshape(-1, D, D)[i], s) for i in range(E * E)])
        iRs = iRs.reshape(E, E, D, D)
        Q = jnp.exp(k[:, None, :, None] + k[None, :, None, :] +
                    maha(ii, -ij, iRs / 2))

        S = jnp.einsum('ji,iljk,kl->il', beta, Q, beta)
        tr = jnp.array([jnp.sum(Q[i, i] * self.iK[i]) for i in range(E)])
        S = (S - jnp.diag(tr)) * t + jnp.diag(jnp.exp(2 * X[:, D]))
        S = S - jnp.outer(M, M)

        return M, S, V
