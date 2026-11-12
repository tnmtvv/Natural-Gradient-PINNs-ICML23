"""
ENGD with PSI inverse-factor approximation (arXiv:2510.19349).
Uses (G + eps I)^(-1), rather than the original least-squares pseudoinverse.
The l2 option changes the metric; energy preserves the original Gramian.
Two dimensional Poisson equation example. Solution given by

u(x,y) = sin(pi*x) * sin(py*y).

"""
import argparse
import math
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit

from ngrad.models import init_params, mlp
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from ngrad.utility import laplace, grid_line_search_factory
from ngrad.inner import model_laplace, model_identity
from jax.flatten_util import ravel_pytree
from adagram_block_factor import block_inverse_factor, factor_direction

jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description="Poisson ENGD with PSI inverse-factor approximation")
parser.add_argument("--rank", type=int, default=32)
parser.add_argument("--eps", type=float, default=1e-6)
parser.add_argument("--block-size", type=int, default=32)
parser.add_argument("--metric", choices=("energy", "l2"), default="energy")
parser.add_argument("--iters", type=int, default=51)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.rank < 1 or args.block_size < 1 or args.iters < 0 or not math.isfinite(args.eps) or args.eps <= 0:
    parser.error("rank/block-size must be positive, iters nonnegative, eps finite and positive")
seed = args.seed

# domains
interior = Square(1.)
boundary = SquareBoundary(1.)

# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
boundary_integrator = DeterministicIntegrator(boundary, 30)
eval_integrator = DeterministicIntegrator(interior, 200)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 32, 1]
params = init_params(layer_sizes, random.PRNGKey(seed))
model = mlp(activation)
v_model = vmap(model, (None, 0))

# solution
@jit
def u_star(x):
    return jnp.prod(jnp.sin(jnp.pi * x))

# rhs
@jit
def f(x):
    return 2. * jnp.pi**2 * u_star(x)

# Differentiate u_theta first, then apply the energy transformation.
# J.T @ J equals the original discretized Gramian for metric="energy".
def tangent(params, x):
    return grad(model)(params, x)

transform = model_laplace if args.metric == "energy" else model_identity

def interior_row(params, x):
    return ravel_pytree(transform(
        lambda z: model(params, z), lambda z: tangent(params, z))(x))[0]

def boundary_row(params, x):
    return ravel_pytree(tangent(params, x))[0]

@jit
def weighted_jacobian(params):
    interior_rows = vmap(interior_row, (None, 0))(params, interior_integrator._x)
    boundary_rows = vmap(boundary_row, (None, 0))(params, boundary_integrator._x)
    return jnp.concatenate((
        jnp.sqrt(interior.measure() / interior_rows.shape[0]) * interior_rows,
        jnp.sqrt(boundary.measure() / boundary_rows.shape[0]) * boundary_rows,
    ))

@jit
def nat_grad(params, grads):
    J = weighted_jacobian(params)
    # Start from eps*I at every optimization step; consume only current J.
    P, Q = block_inverse_factor(J, args.rank, args.eps, args.block_size, "psi")
    g, unravel = ravel_pytree(grads)
    return unravel(factor_direction(g, P, Q, args.eps))

# compute residual
laplace_model = lambda params: laplace(lambda x: model(params, x))
residual = lambda params, x: (laplace_model(params)(x) + f(x))**2.
v_residual =  jit(vmap(residual, (None, 0)))

# loss
@jit
def interior_loss(params):
    return interior_integrator(lambda x: v_residual(params, x))

@jit
def boundary_loss(params):
    return boundary_integrator(lambda x: v_model(params, x)**2)

@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params)

# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(
        lambda x: jnp.dot(grad(error)(x), grad(error)(x))**0.5
        )

def l2_norm(f, integrator):
    return integrator(lambda x: (f(x))**2)**0.5    
   
# natural gradient descent with line search
for iteration in range(args.iters):
    grads = grad(loss)(params)
    nat_grads = nat_grad(params, grads)
    params, actual_step = ls_update(params, nat_grads)
    
    if iteration % 5 == 0:
        # errors
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
    
        print(
            f'NG Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error} and error H1: {h1_error} and step: {actual_step}'
        )
