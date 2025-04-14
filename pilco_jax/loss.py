import jax.numpy as jnp
from jax.numpy import exp, sqrt
from jax.numpy.linalg import solve, det


class Loss:
    def loss_sat(self, m, s):
        D = m.shape[-1] if len(m.shape) > 1 else m.shape[0]

        W = self.W if hasattr(self, 'W') else jnp.eye(D)
        z = self.z if hasattr(self, 'z') else jnp.zeros(D)
        m = jnp.atleast_2d(m)
        z = jnp.atleast_2d(z)

        sW = s @ W
        ispW = solve((jnp.eye(D) + sW).T, W.T).T
        diff = m - z
        L = -exp(-0.5 * (diff @ ispW @ diff.T)) / sqrt(det(jnp.eye(D) + sW))

        i2spW = solve((jnp.eye(D) + 2 * sW).T, W.T).T
        r2 = exp(-1.0 * (diff @ i2spW @ diff.T)) / sqrt(det(jnp.eye(D) + 2 * sW))
        S = r2 - L**2

        t = jnp.dot(W, z.T) - ispW @ (jnp.dot(sW, z.T) + m.T)
        C = L * t

        return L + 1.0, S, C
