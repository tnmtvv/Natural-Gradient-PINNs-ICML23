"""Explicit-batch versions of the repository's three non-Poisson-2D examples.

The analytic spatial derivatives below are exact for the repository's shallow
tanh networks. Parameter derivatives still use JAX autodiff. No training runs
on import. In particular, changing a 5D batch cannot be hidden by a JIT closure.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from ngrad.domains import Interval, Square, SquareBoundary
from ngrad.models import init_params

jax.config.update("jax_enable_x64", True)


@dataclass
class Problem:
    name: str
    layers: list
    initial: object
    unravel: object
    model: object
    spatial: object
    exact: object
    loss: object
    jacobian: object
    metrics: object
    batch: object
    metadata: dict


def build_problem(name, *, small=False):
    """Use repository quadratures; small=True is only for correctness checks."""
    dim, width = (1, 32) if name == "nonlinear" else (5 if name == "poisson5d" else 2, 64)
    layers = [dim, width, 1]
    initial, unravel = ravel_pytree(init_params(layers, jax.random.PRNGKey(0)))

    def spatial(theta, x):
        (w, b), (a, c) = unravel(theta)
        h = jnp.tanh(w @ x + b)
        dh = 1 - h * h
        u = jnp.dot(a[0], h) + c[0]
        du = (a[0] * dh) @ w
        diagonal = (-2 * a[0] * h * dh) @ (w * w)
        return u, du, diagonal

    def model(theta, x):
        (w, b), (a, c) = unravel(theta)
        return jnp.dot(a[0], jnp.tanh(w @ x + b)) + c[0]

    if name == "heat":
        exact = lambda x: jnp.exp(-jnp.pi**2 * x[0] / 4) * jnp.sin(jnp.pi * x[1])
        op = lambda t, x: spatial(t, x)[1][0] - .25 * spatial(t, x)[2][1]
        n, ne = (8, 15) if small else (30, 300)
        xi = Square(1.).deterministic_integration_points(n)
        # Same order and endpoint convention as engd_heat.py.
        xb = jnp.concatenate([SquareBoundary(1., side_number=s).deterministic_integration_points(n)
                              for s in (0, 2, 3)])
        wi = jnp.full(len(xi), 1. / len(xi))
        wb = jnp.full(len(xb), 3. / len(xb))
        target = jnp.concatenate([jnp.zeros(2 * (n - 1)), jnp.sin(jnp.pi * xb[2 * (n - 1):, 1])])
        fixed = (xi, xb, wi, wb, target)
        batch = lambda iteration: fixed
        xe = Square(1.).deterministic_integration_points(ne)
        we = jnp.full(len(xe), 1. / len(xe))
    elif name == "nonlinear":
        exact = lambda x: jnp.cos(jnp.pi * x[0])
        n, ne = (32, 64) if small else (20000, 200000)
        xi = Interval(-1., 1.).deterministic_integration_points(n)
        dx = xi[1, 0] - xi[0, 0]
        wi = jnp.full(n, dx).at[0].multiply(.5).at[-1].multiply(.5)
        fixed = (xi, jnp.empty((0, 1)), wi, jnp.empty(0), jnp.empty(0))
        batch = lambda iteration: fixed
        xe = Interval(-1., 1.).deterministic_integration_points(ne)
        dx_eval = xe[1, 0] - xe[0, 0]
        we = jnp.full(ne, dx_eval).at[0].multiply(.5).at[-1].multiply(.5)
    elif name == "poisson5d":
        exact = lambda x: jnp.sum(jnp.sin(jnp.pi * x))
        op = lambda t, x: jnp.sum(spatial(t, x)[2])
        ni, nb, ne = (40, 20, 100) if small else (4000, 500, 40000)

        @jax.jit
        def batch(iteration):
            # Common random numbers across methods and initialization seeds.
            # Separate evaluation key avoids overlap with the first train batch.
            key = jax.random.fold_in(jax.random.PRNGKey(1729), iteration)
            ki, kb, kd, ks = jax.random.split(key, 4)
            xi = jax.random.uniform(ki, (ni, 5), dtype=jnp.float64)
            xb = jax.random.uniform(kb, (nb, 5), dtype=jnp.float64)
            coordinate = jax.random.randint(kd, (nb,), 0, 5)
            side = jax.random.randint(ks, (nb,), 0, 2)
            xb = xb.at[jnp.arange(nb), coordinate].set(side.astype(jnp.float64))
            return xi, xb, jnp.full(ni, 1. / ni), jnp.full(nb, 10. / nb), jax.vmap(exact)(xb)

        xe = jax.random.uniform(jax.random.PRNGKey(271828), (ne, 5), dtype=jnp.float64)
        we = jnp.full(ne, 1. / ne)
    else:
        raise ValueError(name)

    vm = jax.vmap(model, (None, 0))
    vsp = jax.vmap(spatial, (None, 0))
    value_rows = jax.vmap(jax.grad(model), (None, 0))

    def loss(theta, data):
        xi, xb, wi, wb, target = data
        if name == "nonlinear":
            u, du, _ = vsp(theta, xi)
            us = jax.vmap(exact)(xi)
            rhs = jnp.pi**2 * us + us**3
            return jnp.dot(wi, .5 * du[:, 0]**2 + .25 * u**4 - rhs * u)
        residual = jax.vmap(op, (None, 0))(theta, xi)
        if name == "poisson5d":
            residual += jnp.pi**2 * jax.vmap(exact)(xi)
        value = jnp.dot(wi, residual**2) + jnp.dot(wb, (vm(theta, xb) - target)**2)
        return .5 * value if name == "poisson5d" else value

    def jacobian(theta, data):
        xi, xb, wi, wb, _ = data
        if name == "nonlinear":
            derivative = lambda t, x: spatial(t, x)[1][0]
            dx_rows = jax.vmap(jax.grad(derivative), (None, 0))(theta, xi)
            # D²E(u)(v,w) = int v'w' + 3u²vw. This is NOT a residual Jacobian.
            nonlinear_rows = jnp.sqrt(3.) * vm(theta, xi)[:, None] * value_rows(theta, xi)
            return jnp.concatenate([jnp.sqrt(wi)[:, None] * dx_rows,
                                    jnp.sqrt(wi)[:, None] * nonlinear_rows])
        interior_rows = jax.vmap(jax.grad(op), (None, 0))(theta, xi)
        return jnp.concatenate([jnp.sqrt(wi)[:, None] * interior_rows,
                                jnp.sqrt(wb)[:, None] * value_rows(theta, xb)])

    exact_values = jax.vmap(exact)(xe)
    exact_grad = jax.vmap(jax.grad(exact))(xe)
    norm_l2 = jnp.sqrt(jnp.dot(we, exact_values**2))
    norm_grad = jnp.sqrt(jnp.dot(we, jnp.sum(exact_grad**2, axis=1)))

    @jax.jit
    def metrics(theta):
        u, du, _ = vsp(theta, xe)
        error_l2 = jnp.sqrt(jnp.dot(we, (u - exact_values)**2))
        error_grad = jnp.sqrt(jnp.dot(we, jnp.sum((du - exact_grad)**2, axis=1)))
        # Preserve the repository's sum-of-seminorms H1 convention.
        return jnp.array([error_l2 / norm_l2, (error_l2 + error_grad) / (norm_l2 + norm_grad)])

    first = batch(0)
    metadata = dict(name=name, layers=layers, parameters=len(initial),
                    interior_points=len(first[0]), boundary_points=len(first[1]),
                    evaluation_points=len(xe), small=small,
                    resampling="explicit fresh common batch each update" if name == "poisson5d" else "fixed quadrature",
                    h1="(L2(error)+L2(gradient error))/(L2(exact)+L2(exact gradient))",
                    normalization="repository loss and Gram (heat Gram omits common Hessian factor 2)")
    return Problem(name, layers, initial, unravel, model, spatial, exact,
                   jax.jit(loss), jax.jit(jacobian), metrics, batch, metadata)
