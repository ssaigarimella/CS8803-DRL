import jax.numpy as jnp
from jax.numpy import log
from pilco_jax.util import fill_mat

def congp(policy, m, s):
    policy.hyp = jnp.atleast_2d(policy.p['hyp'])
    policy.inputs = jnp.atleast_2d(policy.p['inputs'])
    policy.targets = jnp.atleast_2d(policy.p['targets'])

    T = jnp.zeros_like(policy.hyp)
    log_bounds = jnp.array([log(1.0), log(0.01)])
    tiled = jnp.tile(log_bounds, (policy.hyp.shape[0], 1))
    T = T.at[:, -2:].set(tiled)
    mask = jnp.array(T == 0, dtype=bool)
    policy.hyp = jnp.where(mask, policy.hyp, T)

    return policy.gp2(m, s)

def concat(con, sat, policy, m, s):
    max_u = jnp.array(policy.max_u)
    E = len(max_u)
    D = len(m)

    F = D + E
    i = jnp.arange(D)
    j = jnp.arange(D, F)
    M = m
    S = fill_mat(s, jnp.zeros((F, F)))

    m, s, c = con(policy, m, s)
    M = jnp.hstack([M, m])
    S = fill_mat(s, S, j, j)
    q = S[jnp.ix_(i, i)] @ c
    S = fill_mat(q, S, i, j)
    S = fill_mat(q.T, S, j, i)

    M, S, R = sat(M, S, j, max_u)
    C = jnp.hstack([jnp.eye(D), c]) @ R
    return M, S, C
