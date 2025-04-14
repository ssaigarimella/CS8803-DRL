import jax.numpy as jnp
from jax import value_and_grad
from jax.numpy import newaxis
from jax.numpy import array as jnp_array
import jax.random as random
from jax.random import multivariate_normal, uniform
from jax.experimental.ode import odeint
from scipy.optimize import minimize

from pilco_jax.util import gaussian_trig, fill_mat, unwrap, rewrap

def rollout(start, policy, plant, cost, H, key):
    """
    Generate a state trajectory using an ODE solver.
    """
    odei = plant.odei
    poli = plant.poli
    dyno = plant.dyno
    angi = plant.angi

    odei_idx = jnp.array(odei)
    poli_idx = jnp.array(poli)
    dyno_idx = jnp.array(dyno)

    nX = len(odei)
    nU = len(policy.max_u)
    nA = len(angi)

    state = start
    x = jnp.zeros([H + 1, nX + 2 * nA])
    # x = x.at[0, odei_idx].set(multivariate_normal(start, plant.noise))
    
    key, subkey1 = random.split(key)
    x = x.at[0, odei_idx].set(multivariate_normal(subkey1, start, plant.noise))

    u = jnp.zeros([H, nU])
    y = jnp.zeros([H, nX])
    L = jnp.zeros(H)
    latent = jnp.zeros([H + 1, nX + nU])

    for i in range(H):
        s = x[i, odei_idx]
        a, _, _ = gaussian_trig(s, 0 * jnp.eye(nX), angi)
        s = jnp.hstack([s, a])
        x = x.at[i, -2 * nA:].set(s[-2 * nA:])

        if hasattr(policy, "fcn"):
            u_i, _, _ = policy.fcn(s[poli_idx], 0 * jnp.eye(len(poli)))
            u = u.at[i, :].set(u_i)
        else:
            key, subkey_rand = random.split(key)
            u = u.at[i, :].set(jnp.array(policy.max_u) * (2 * uniform(subkey_rand, shape=(nU,)) - 1))

        latent = latent.at[i, :].set(jnp.hstack([state, u[i, :]]))

        dynamics = plant.dynamics
        dt = plant.dt
        def dynamics_wrapped(z, t):
            return dynamics(z, t, u[i, :])

        next_state = odeint(dynamics_wrapped, state[odei_idx], jnp.array([0.0, dt]))
        state = next_state[-1, :]
        # x = x.at[i + 1, odei_idx].set(multivariate_normal(state, plant.noise))
        
        key, subkey2 = random.split(key)
        x = x.at[i + 1, odei_idx].set(multivariate_normal(subkey2, state, plant.noise))

        if hasattr(cost, "fcn"):
            L = L.at[i].set(cost.fcn(state[dyno_idx], 0 * jnp.eye(len(dyno))).reshape(()))

    y = x[1:H + 1, :nX]
    x = jnp.hstack([x[:H, :], u[:H, :]])
    latent = latent.at[H, :nX].set(state)

    return x, y, L, latent, key

def train(gpmodel, plant, policy, x, y):
    Du = len(policy.max_u)
    dyni = jnp.array(plant.dyni)
    dyno = jnp.array(plant.dyno)
    difi = jnp.array(plant.difi)

    gpmodel.inputs = jnp.hstack([x[:, dyni], x[:, -Du:]])
    gpmodel.targets = y[:, dyno]
    gpmodel.targets = gpmodel.targets.at[:, difi].set(gpmodel.targets[:, difi] - x[:, dyno[difi]])
    gpmodel.optimize()

    hyp = gpmodel.hyp
    print(gpmodel.result['message'])
    print("Learned noise std:\n%s" % (str(jnp.exp(hyp[:, -1]))))
    print("SNRs:\n%s" % (str(jnp.exp(hyp[:, -2] - hyp[:, -1]))))

def propagate(m, s, plant, dynmodel, policy):
    angi = plant.angi
    poli = plant.poli
    dyni = plant.dyni
    difi = plant.difi

    D0 = len(m)
    D1 = D0 + 2 * len(angi)
    D2 = D1 + len(policy.max_u)
    M = jnp.array(m)
    S = s

    i, j = jnp.arange(D0), jnp.arange(D0, D1)
    m, s, c = gaussian_trig(M[i], S[jnp.ix_(i, i)], angi)
    q = jnp.matmul(S[jnp.ix_(i, i)], c)
    M = jnp.hstack([M, m])
    S = jnp.vstack([jnp.hstack([S, q]), jnp.hstack([q.T, s])])

    i, j = jnp.array(poli), jnp.arange(D1)
    m, s, c = policy.fcn(M[i], S[jnp.ix_(i, i)])
    q = jnp.matmul(S[jnp.ix_(j, i)], c)
    M = jnp.hstack([M, m])
    S = jnp.vstack([jnp.hstack([S, q]), jnp.hstack([q.T, s])])

    i, j = jnp.hstack([dyni, jnp.arange(D1, D2)]), jnp.arange(D2)
    m, s, c = dynmodel.fcn(M[i], S[jnp.ix_(i, i)])
    q = jnp.matmul(S[jnp.ix_(j, i)], c)
    M = jnp.hstack([M, m])
    S = jnp.vstack([jnp.hstack([S, q]), jnp.hstack([q.T, s])])

    P = jnp.hstack([jnp.zeros((D0, D2)), jnp.eye(D0)])
    P = fill_mat(jnp.eye(len(difi)), P, difi, difi)
    M_next = jnp.matmul(P, M[:, newaxis]).flatten()
    S_next = P @ S @ P.T
    S_next = (S_next + S_next.T) / 2
    return M_next, S_next

def value(p, mu0, S0, dynmodel, policy, plant, cost, H):
    policy.p = rewrap(p, policy.p)

    M = mu0
    S = S0
    L = jnp.array([0.0])
    for t in range(H):
        M, S = plant.prop(M, S, plant, dynmodel, policy)
        L = L + cost.gamma**t * cost.fcn(M, S)
    # return L
    return L.reshape(())


def learn(mu0, S0, dynmodel, policy, plant, cost, H):
    global num_iters
    num_iters = 0
    args = (mu0, S0, dynmodel, policy, plant, cost, H)
    options = {'maxiter': 10, 'disp': True}

    def callback(p):
        L = float(value(p, *args))
        global num_iters
        num_iters += 1
        print("linesearch %d: %s" % (num_iters, str(L)))

    print("Perform policy searching...")
    result = minimize(
        value_and_grad(value),
        unwrap(policy.p),
        args,
        jac=True,
        options=options,
        callback=callback)

    policy.p = rewrap(result.get('x'), policy.p)
    return result
