import jax.numpy as jnp
from jax.numpy import sin, cos, exp

__all__ = ['Empty']


class Empty(dict):
    pass


def fill_mat(m, n, i=None, j=None):
    m, n = jnp.atleast_2d(m), jnp.atleast_2d(n)
    a, b = m.shape
    p, q = n.shape
    i = jnp.arange(a) if i is None else jnp.atleast_1d(i)
    j = jnp.arange(b) if j is None else jnp.atleast_1d(j)

    if a > p or b > q:
        raise ValueError("Shape error!")
    if len(i) != a or len(j) != b:
        raise ValueError("Indices not match!")
    if not (jnp.max(i) < p and jnp.max(j) < q):
        raise ValueError("Indices out of bound!")

    Ti = jnp.zeros((p, a)).at[i, jnp.arange(a)].set(1)
    Tj = jnp.zeros((b, q)).at[jnp.arange(b), j].set(1)
    return Ti @ m @ Tj + n


def gaussian_trig(m, v, i, e=None):
    i = jnp.array(i)
    d = len(m)
    L = len(i)
    e = jnp.ones((1, L)) if e is None else jnp.atleast_2d(e)
    ee = jnp.vstack([e, e]).reshape(1, -1, order='F')

    mi = jnp.atleast_2d(m[i])
    vi = v[jnp.ix_(i, i)]
    vii = jnp.atleast_2d(jnp.diag(vi))

    M = jnp.vstack([e * exp(-vii / 2) * sin(mi), e * exp(-vii / 2) * cos(mi)])
    M = M.flatten(order='F')

    lq = -(vii.T + vii) / 2
    q = exp(lq)

    U1 = (exp(lq + vi) - q) * sin(mi.T - mi)
    U2 = (exp(lq - vi) - q) * sin(mi.T + mi)
    U3 = (exp(lq + vi) - q) * cos(mi.T - mi)
    U4 = (exp(lq - vi) - q) * cos(mi.T + mi)

    V = jnp.vstack([
        jnp.hstack([U3 - U4, U1 + U2]),
        jnp.hstack([(U1 + U2).T, U3 + U4])
    ])
    V = jnp.vstack([
        jnp.hstack([V[::2, ::2], V[::2, 1::2]]),
        jnp.hstack([V[1::2, ::2], V[1::2, 1::2]])
    ])
    V = jnp.dot(ee.T, ee) * V / 2

    C = jnp.hstack([jnp.diag(M[1::2]), -jnp.diag(M[::2])])
    C = jnp.hstack([C[:, ::2], C[:, 1::2]])
    C = fill_mat(C, jnp.zeros((d, 2 * L)), i, None)

    return M, V, C


def gaussian_sin(m, v, i, e=None):
    i = jnp.array(i)
    d = len(m)
    L = len(i)
    e = jnp.ones((1, L)) if e is None else jnp.atleast_2d(e)

    mi = jnp.atleast_2d(m[i])
    vi = v[jnp.ix_(i, i)]
    vii = jnp.atleast_2d(jnp.diag(vi))
    M = e * exp(-vii / 2) * sin(mi)
    M = M.flatten()

    lq = -(vii.T + vii) / 2
    q = exp(lq)
    V = ((exp(lq + vi) - q) * cos(mi.T - mi) -
         (exp(lq - vi) - q) * cos(mi.T + mi))
    V = jnp.dot(e.T, e) * V / 2

    C = jnp.diag((e * exp(-vii / 2) * cos(mi)).flatten())
    C = fill_mat(C, jnp.zeros((d, L)), i, None)

    return M, V, C


def maha(a, b, Q):
    aQ = jnp.matmul(a, Q)
    bQ = jnp.matmul(b, Q)
    K = jnp.expand_dims(jnp.sum(aQ * a, -1), -1) + jnp.expand_dims(
        jnp.sum(bQ * b, -1), -2) - 2 * jnp.einsum('...ij, ...kj->...ik', aQ, b)
    return K


def unwrap(p):
    return jnp.hstack([v.flatten() for v in p.values()])


def rewrap(m, p):
    d = {}
    start = 0
    for k, v in p.items():
        length = jnp.size(v)
        d[k] = jnp.reshape(m[start:start + length], v.shape)
        start = start + length
    return d