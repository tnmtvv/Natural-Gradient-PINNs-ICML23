"""Compare optimizer-induced function changes with the known Poisson solution."""
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from reproduce_poisson_fig1 import build
from ngrad.models import init_params
from ngrad.domains import Square
from ngrad.inner import model_laplace
from adagram_block_factor import block_inverse_factor, factor_direction

jax.config.update('jax_enable_x64', True)


def alignment(v, e):
    """Uniform quadrature on the unit square; zero vectors have undefined angles."""
    vn = jnp.sqrt(jnp.mean(v*v))
    en = jnp.sqrt(jnp.mean(e*e))
    valid = (vn > 1e-14) & (en > 1e-14)
    cosine = jnp.where(valid, jnp.clip(jnp.mean(v*e)/(vn*en), -1., 1.), jnp.nan)
    return jnp.degrees(jnp.arccos(cosine)), vn


def ridge_cut_direction(J, g, eps):
    """Ridge inverse on modes retained by the default E-NGD Gram cutoff.

    Use an SVD of J to avoid forming a poorly conditioned Gramian. For
    G = J.T @ J, the default lstsq threshold is p * machine_eps * lambda_max.
    Recompute this threshold at every iteration, rather than fixing a rank.
    """
    _, singular, Vt = jnp.linalg.svd(J, full_matrices=False)
    eigenvalues = singular**2
    cutoff = J.shape[1] * jnp.finfo(J.dtype).eps * eigenvalues[0]
    keep = (eigenvalues > 0) & (eigenvalues >= cutoff)
    coefficients = jnp.where(keep, Vt @ g, 0.) / (eigenvalues + eps)
    return Vt.T @ coefficients, jnp.sum(keep), cutoff


def ridge_rank_direction(J, g, eps, rank):
    """Keep exactly the largest `rank` singular modes of J; zero the rest.

    This is a spectral rank constraint, distinct from the low-rank inverse
    factor correction in block_inverse_factor. A full SVD is used here so
    these diagnostics measure the filter without an approximation error.
    """
    if not 1 <= rank <= min(J.shape):
        raise ValueError('spectral rank must be between 1 and min(J.shape)')
    _, singular, Vt = jnp.linalg.svd(J, full_matrices=False)
    basis = Vt[:rank]
    coefficients = (basis @ g) / (singular[:rank]**2 + eps)
    return basis.T @ coefficients, jnp.asarray(rank), singular[rank-1]**2


def run(ctx, params, rank, block, compression, args, return_final=False, checkpoint_callback=None):
    model = ctx['model']
    points = Square(1.).deterministic_integration_points(97)
    exact = jnp.prod(jnp.sin(jnp.pi*points), axis=1)
    predict = jax.jit(lambda p: jax.vmap(model, (None, 0))(p, points))

    def row(p, x):
        return ravel_pytree(model_laplace(lambda z: model(p,z),
            lambda z: jax.grad(model)(p,z))(x))[0]

    @jax.jit
    def step(p):
        g, unravel = ravel_pytree(jax.grad(ctx['loss'])(p))
        J = jnp.concatenate((jnp.sqrt(ctx['interior_weight'])*jax.vmap(row,(None,0))(p,ctx['interior_points']),
            jnp.sqrt(ctx['boundary_weight'])*jax.vmap(lambda x: ravel_pytree(jax.grad(model)(p,x))[0])(ctx['boundary_points'])))
        if rank is None:
            d, _, retained_modes, singular = jnp.linalg.lstsq(J.T@J,g,rcond=None)
            cutoff = J.shape[1] * jnp.finfo(J.dtype).eps * singular[0]
        elif compression == 'svd_cut':
            d, retained_modes, cutoff = ridge_cut_direction(J, g, args.eps)
        elif compression == 'svd_rank':
            d, retained_modes, cutoff = ridge_rank_direction(J, g, args.eps, rank)
        else:
            P,Q = block_inverse_factor(J,rank,args.eps,block,compression)
            d = factor_direction(g,P,Q,args.eps)
            retained_modes, cutoff = jnp.asarray(J.shape[1]), jnp.asarray(0.)
        next_p, eta = ctx['ls_update'](p,unravel(d))
        return next_p, eta, d, jnp.linalg.norm(g), retained_modes, cutoff

    @jax.jit
    def diagnose(p, next_p, d, eta):
        _, unravel = ravel_pytree(p)
        u, tangent = jax.jvp(predict,(p,),(unravel(-d),))
        e = exact-u
        actual = predict(next_p)-u
        angle, norm = alignment(tangent,e)
        actual_angle, actual_norm = alignment(actual,e)
        error = jnp.sqrt(jnp.mean(e*e))
        rel = error/jnp.sqrt(jnp.mean(exact*exact))
        values = jnp.array([rel,angle,actual_angle,norm,eta*norm,actual_norm,error,jnp.sqrt(jnp.mean((actual-eta*tangent)**2))])
        if getattr(args, 'gradient_alignment', False):
            negative_gradient = jax.tree_util.tree_map(lambda x: -x, jax.grad(ctx['loss'])(p))
            _, gradient_tangent = jax.jvp(predict, (p,), (negative_gradient,))
            gradient_angle, gradient_velocity = alignment(gradient_tangent, e)
            values = jnp.concatenate((values, jnp.array([gradient_angle, gradient_velocity])))
        if getattr(args, 'parameter_alignment', False):
            g = ravel_pytree(jax.grad(ctx['loss'])(p))[0]
            gn, dn = jnp.linalg.norm(g), jnp.linalg.norm(d)
            valid = (gn > 1e-14) & (dn > 1e-14)
            cosine = jnp.where(valid, jnp.clip(jnp.dot(g / gn, d / dn), -1., 1.), jnp.nan)
            values = jnp.concatenate((values, jnp.array([jnp.degrees(jnp.arccos(cosine)), ctx['loss'](p)])))
        if getattr(args, 'cross_direction_alignment', False):
            g = ravel_pytree(jax.grad(ctx['loss'])(p))[0]
            J = jnp.concatenate((jnp.sqrt(ctx['interior_weight'])*jax.vmap(row,(None,0))(p,ctx['interior_points']),
                jnp.sqrt(ctx['boundary_weight'])*jax.vmap(lambda x: ravel_pytree(jax.grad(model)(p,x))[0])(ctx['boundary_points'])))
            # Evaluate the counterfactual solver at the current trajectory point.
            if rank is None:
                P, Q = block_inverse_factor(J, J.shape[1], args.cross_eps, block, 'svd')
                other = factor_direction(g, P, Q, args.cross_eps)
            else:
                other = jnp.linalg.lstsq(J.T @ J, g, rcond=None)[0]
            dn, on = jnp.linalg.norm(d), jnp.linalg.norm(other)
            a, b = d / dn, other / on
            cross_angle = jnp.degrees(2*jnp.arctan2(jnp.linalg.norm(a-b), jnp.linalg.norm(a+b)))
            cross_angle = jnp.where((dn > 1e-14) & (on > 1e-14), cross_angle, jnp.nan)
            values = jnp.concatenate((values, jnp.array([cross_angle, on])))
        return values

    rows=[]
    for i in range(args.iters):
        next_p, eta, d, gn, retained_modes, cutoff = jax.block_until_ready(step(params))
        if i % args.log_every == 0 or i == args.iters-1:
            values=np.asarray(jax.block_until_ready(diagnose(params,next_p,d,eta)))
            if not np.all(np.isfinite(values[[0, 3, 4, 5, 6, 7]])):
                raise FloatingPointError(f'Nonfinite function diagnostic at iteration {i}')
            rows.append([i,*values,float(eta),float(jnp.linalg.norm(d)),float(gn),int(retained_modes),float(cutoff)])
            print(f'{compression} rank={rank} block={block}: {i} rel={values[0]:.3e} angle={values[1]:.2f} actual={values[2]:.2f} eta={float(eta):.3e} modes={int(retained_modes)}',flush=True)
        if checkpoint_callback is not None:
            checkpoint_callback(i, params, next_p, d, eta)
        params=next_p
    history = np.asarray(rows)
    return (history, params) if return_final else history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iters',type=int,default=800)
    parser.add_argument('--log-every',type=int,default=10)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--eps',type=float,default=1e-6)
    parser.add_argument('--method',choices=('both', 'svd', 'engd', 'svd-cut', 'svd-rank'),default='both')
    parser.add_argument('--spectral-rank',type=int,nargs='+',
                        help='One or more fixed spectral ranks; requires --method svd-rank')
    parser.add_argument('--block-size',type=int,default=32)
    parser.add_argument('--out-dir',type=Path,default=Path('results/function_diagnostics'))
    args=parser.parse_args()
    if args.iters<1 or args.log_every<1 or args.block_size<1 or not np.isfinite(args.eps) or args.eps<=0:
        parser.error('iters, log-every, block-size and eps must be positive')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    ctx=build()
    params=init_params(ctx['layer_sizes'],jax.random.PRNGKey(args.seed))
    size=ravel_pytree(params)[0].size
    if args.method == 'svd-rank':
        if not args.spectral_rank or any(not 1 <= r <= size for r in args.spectral_rank):
            parser.error(f'--method svd-rank requires --spectral-rank values between 1 and {size}')
    elif args.spectral_rank is not None:
        parser.error('--spectral-rank requires --method svd-rank')
    configs=[]
    if args.method in ('both', 'engd'):
        configs.append((None,args.block_size,'engd'))
    if args.method in ('both', 'svd'):
        configs.append((size,args.block_size,'svd'))
    if args.method == 'svd-cut':
        configs.append((size,args.block_size,'svd_cut'))
    if args.method == 'svd-rank':
        configs.extend((r,args.block_size,'svd_rank') for r in args.spectral_rank)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(14,12))
    columns=['iteration','relative_l2','linear_angle_deg','actual_angle_deg','velocity_l2','linear_step_l2','actual_step_l2','error_l2','linearization_error_l2','eta','direction_norm','gradient_norm','retained_modes','eigenvalue_cutoff']
    for rank,block,compression in configs:
        name='engd' if rank is None else f'{compression}_rank{rank}_block{block}'
        data,final_params=run(ctx,params,rank,block,compression,args,return_final=True)
        final_relative_l2=float(ctx['rel_l2'](final_params))
        if not np.isfinite(final_relative_l2):
            raise FloatingPointError('Nonfinite final relative L2 error')
        spectral_selection = {'svd_cut': 'engd_default', 'svd_rank': 'fixed_rank'}.get(compression, 'none')
        solver = {'svd_cut': 'jacobian_svd_ridge_cut', 'svd_rank': 'jacobian_svd_ridge_rank'}.get(compression, 'gram_lstsq' if rank is None else 'block_inverse_factor')
        np.savez(args.out_dir/f'{name}.npz',**dict(zip(columns,data.T)),seed=args.seed,eps=0. if rank is None else args.eps, rank=size if rank is None else rank, parameter_count=size, block_size=block, jax_version=jax.__version__, eval_points=95**2, angle_norm_threshold=1e-14, final_iteration=args.iters, final_relative_l2=final_relative_l2, final_theta=np.asarray(ravel_pytree(final_params)[0]), spectral_cutoff=spectral_selection, direction_solver=solver)
        print(f'{name}: after {args.iters} updates, relative L2 {final_relative_l2:.9e}',flush=True)
        label=name.replace('_',' ')
        for ax,col,title in zip(axs.flat,[1,2,3,4,6,9],['Relative L2 (before step)','Linear angle to exact solution (deg)','Actual angle to exact solution (deg)','Function velocity L2','Actual function step L2','Line search step']):
            ax.plot(data[:,0],data[:,col],label=label)
            ax.set(title=title,xlabel='Iteration')
            if col in [1,4,6,9]: ax.set_yscale('log')
            else: ax.axhline(90,color='gray',ls=':'); ax.set_ylim(0,180)
            ax.grid(True,which='both',alpha=.3)
    fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3)
    fig.tight_layout(rect=(0,.07,1,1))
    fig.savefig(args.out_dir/'diagnostics.png',dpi=160)
    plt.close(fig)

if __name__=='__main__':
    main()
