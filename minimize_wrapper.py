# from lbfgs import fmin_lbfgs
# from lbfgs._lowlevel import LBFGSError
from scipy.optimize import minimize, rosen, rosen_der, fmin_l_bfgs_b
# import tensorflow_probability as tfp

import numpy as np
# import nlopt
# from optimparallel import minimize_parallel

# # from LBFGS import LBFGS, FullBatchLBFGS
# from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS
# import torch
# import torch.nn as nn
# import torch.optim as optim
from fminlbfgs.fminlbfgs import matlab_fmin_lbfgs


def cstm_minimize(fun, jac, x0, options, alg, matlab_options, **kwargs):
    gtol = options["gtol"]
    max_iter = options["maxiter"]
    return cstm_minimize2(fun, jac, x0, gtol, max_iter, alg, matlab_options=matlab_options)


def cstm_minimize2(fun, jac, x0, gtol, max_iter, alg, matlab_options, **kwargs):
    if alg == "scipy":
        res = minimize(fun=fun, jac=jac, method='l-bfgs-b', x0=x0, options={'gtol': gtol, 'maxiter': max_iter})
        return res.x, res.nit, res.fun
    elif alg == "scipy-nojac":
        res = minimize(fun=fun, method='l-bfgs-b', x0=x0, options={'gtol': gtol, 'maxiter': max_iter})
        return res.x, res.nit
    elif alg == "matlab":
        def opt_proxy(rs_vars):
            if callable(jac):
                cost = fun(rs_vars.reshape(-1))
                grad = jac(rs_vars.reshape(-1)).reshape(rs_vars.shape)
                return cost, grad
            elif jac:
                cost, all_grads = fun(rs_vars)
                return cost, all_grads

        x, fval, exitfalg, output, grad, nit = matlab_fmin_lbfgs(opt_proxy, x0.reshape(-1), matlab_options)  # TODO optim parameter
        return x, nit, fval
    elif alg == "fmin_l_bfgs_b":
        (x, _, d) = fmin_l_bfgs_b(fun, x0, jac, maxiter=max_iter)
        return x, d["nit"]
    elif alg == "pylbfgs":
        def opt_proxy(rs_vars, fill_grad):
            if callable(jac):
                cost = fun(rs_vars)
                fill_grad[:] = jac(rs_vars).reshape(rs_vars.shape)
                return cost
            elif jac:
                cost, all_grads = fun(rs_vars)
                fill_grad[:] = all_grads
                return cost

        nit = -1
        prev_x = None

        def pg(x, g, fx, xnorm, gnorm, step, k, num_eval, *args):
            nonlocal nit, prev_x
            nit = max(nit, k)
            prev_x = x

        try:
            res_x = fmin_lbfgs(opt_proxy, x0, max_iterations=max_iter, progress=pg, gtol=gtol, ftol=gtol)
            return res_x, nit
        except LBFGSError as e:
            if e.args[0] == 'The algorithm routine reaches the maximum number of iterations.':
                return prev_x, nit
            else:
                raise e

    elif alg == "tensorflow":
        def opt_proxy(rs_vars):
            if callable(jac):
                return tf.constant(fun(rs_vars.numpy())), tf.constant(jac(rs_vars.numpy()).reshape(rs_vars.shape))
            elif jac:
                return tf.constant(fun(rs_vars))

        # TODO gtol
        result = tfp.optimizer.bfgs_minimize(value_and_gradients_function=opt_proxy,
                                             initial_position=x0,
                                             max_iterations=max_iter)

        return result.position.numpy(), result.num_iterations.numpy()
    elif alg == "dlib":
        pass
    elif alg == "nlopt":
        """ Copy paste from NLopt docs: 
        NLopt includes several variations of this algorithm by Prof. Luksan. 
        First, a variant preconditioned by the low-storage BFGS algorithm with steepest-descent restarting, 
        specified as NLOPT_LD_TNEWTON_PRECOND_RESTART. 
        Second, simplified versions NLOPT_LD_TNEWTON_PRECOND (same without restarting), 
        NLOPT_LD_TNEWTON_RESTART (same without preconditioning), 
        and NLOPT_LD_TNEWTON (same without restarting or preconditioning). """

        def opt_proxy(rs_vars, fill_grad):
            if callable(jac):
                cost = fun(rs_vars)
                fill_grad[:] = jac(rs_vars).reshape(rs_vars.shape)
                return cost
            elif jac:
                cost, all_grads = fun(rs_vars)
                fill_grad[:] = all_grads
                return cost

        num_vars = x0.reshape(-1).shape[0]
        print(f"num_vars={num_vars}")
        opt = nlopt.opt(nlopt.LD_LBFGS, num_vars)  # nlopt.LD_TNEWTON_PRECOND
        opt.set_min_objective(opt_proxy)
        opt.set_xtol_rel(1e-4)

        # try:
        xopt = opt.optimize(x0.reshape(-1))
        print(xopt)
        # except Exception as e:
        #     print(e)

        opt_val = opt.last_optimum_value()
        result = opt.last_optimize_result()
        print(opt_val, result)
    elif alg == "optimparallel":
        res = minimize_parallel(fun=fun, jac=jac, x0=x0, options={'gtol': gtol, 'maxiter': max_iter})
        return res.x, res.nit
    else:
        raise ValueError(f"{alg} not in supported algorithms")
    # elif alg == "pytorch":
    #     t_x0 = torch.tensor(x0)
    #     optim = FullBatchLBFGS(t_x0)

# x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
# opts = {'gtol': 1e-6, 'disp': True}
# gtol = 1e-6
# max_iterations = 100
# for alg in ["scipy", "fmin_l_bfgs_b", "pylbfgs", "tensorflow"]:
#     res, iters = cstm_minimize2(fun=rosen, jac=rosen_der, x0=x0, gtol=gtol, max_iter=max_iterations, alg=alg)
#     print(f"{alg}: {iters} -> {res}")
