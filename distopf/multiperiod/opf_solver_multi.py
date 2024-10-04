from collections.abc import Callable, Collection
from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array
from distopf import LinDistModelQ, LinDistModelP, LinDistModel


# cost = pd.read_csv("cost_data.csv")
def gradient_load_min(model):
    c = np.zeros(model.n_x)
    for ph in "abc":
        if model.phase_exists(ph):
            c[model.branches_out_of_j("pij", 0, ph)] = 1
    return c


def gradient_curtail(model):
    c = np.zeros(model.n_x)
    for i in range(
        model.p_der_start_phase_idx["a"],
        model.p_der_start_phase_idx["c"] + len(model.der_bus["c"]),
    ):
        c[i] = -1
    return c


# ~~~ Quadratic objective with linear constraints for use with solve_quad()~~~
# def cp_obj_loss(model, xk):
#     f: cp.Expression = 0
#     for t in range(LinDistModelQ.n):
#         for j in range(1, model.nb):
#             for a in "abc":
#                 if model.phase_exists(a, t, j):
#                     i = model.idx("bi", j, a, t)
#                     f += model.r[a + a][i, j] * (xk[model.idx("pij", j, a, t)[0]] ** 2)
#                     f += model.r[a + a][i, j] * (xk[model.idx("qij", j, a, t)[0]] ** 2)
#                     dis = model.idx("pd", j, a, t)
#                     ch = model.idx("pc", j, a, t)
#                     if ch:
#                         f += 1e-1*(1 - 0.95) * (xk[ch])
#                     if dis:
#                         f += 1e-1*((1/0.95)-1) * (xk[dis])
#     return f


def cp_obj_loss(model: LinDistModel, xk: cp.Variable, **kwargs) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    f_list = []
    for t in range(model.n):
        for j in range(1, model.nb):
            for a in "abc":
                if model.phase_exists(a, t, j):
                    i = model.idx("bi", j, a, t)
                    f_list.append(
                        model.r[a + a][i, j] * (xk[model.idx("pij", j, a, t)[0]] ** 2)
                    )
                    f_list.append(
                        model.r[a + a][i, j] * (xk[model.idx("qij", j, a, t)[0]] ** 2)
                    )
                    if model.battery:
                        dis = model.idx("pd", j, a, t)
                        ch = model.idx("pc", j, a, t)
                        if ch:
                            f_list.append(
                                1e-3 * (1 - model.bat["nc_" + a].get(j, 1)) * (xk[ch])
                            )
                        if dis:
                            f_list.append(
                                1e-3
                                * ((1 / model.bat["nd_" + a].get(j, 1)) - 1)
                                * (xk[dis])
                            )
    return cp.sum(f_list)


# def cp_obj_loss(model, xk):
#     f: cp.Expression = 0
#     for t in range(LinDistModelQ.n):
#         for j in range(1, model.nb):
#             for a in "abc":
#                 if model.phase_exists(a, t, j):
#                     i = model.idx("bi", j, a, t)[0]
#                     f += model.r[a + a][i, j] * (xk[model.idx("pij", j, a, t)[0]] ** 2)
#                     f += model.r[a + a][i, j] * (xk[model.idx("qij", j, a, t)[0]] ** 2)
#                     if LinDistModel.battery:
#                         dis = model.idx("pd", j, a, t)
#                         ch = model.idx("pc", j, a, t)
#                         if ch:
#                             f += 1e-3*(1 - model.bat["nc_" + a].get(j,1)) * (xk[ch])
#                         if dis:
#                             f += 1e-3*((1/model.bat["nd_" + a].get(j,1))-1) * (xk[dis])
#                         # if dis:
#                         #     f += 1e-3*((1/model.bat["nd_" + a].get(j,0))-model.bat["nc_" + a].get(j,0)) * (xk[dis])
#     return f


def peak_shave(model, xk):
    f: cp.Expression = 0
    subs = []
    for t in range(LinDistModelQ.n):
        ph = 0
        for a in "abc":
            ph += xk[model.idx("pij", model.SWING + 1, a, t)[0]]
        subs.append(ph)
        for j in range(1, model.nb):
            for a in "abc":
                if model.phase_exists(a, t, j):
                    if LinDistModelQ.battery:
                        dis = model.idx("pd", j, a, t)
                        ch = model.idx("pc", j, a, t)
                        if ch:
                            f += 1e-3 * (1 - model.bat["nc_" + a].get(j, 1)) * (xk[ch])
                        if dis:
                            f += (
                                1e-3
                                * ((1 / model.bat["nd_" + a].get(j, 1)) - 1)
                                * (xk[dis])
                            )
                        # if dis:
                        #     f += 1e-5*((1/model.bat["nd_" + a].get(j,0))-model.bat["nc_" + a].get(j,0)) * (xk[dis])
    f += cp.max(cp.hstack(subs))
    return f


peak_h = [17, 18, 19, 20, 21]
peak_price = 19
off_peak_price = 7


def cost_min(model, xk):
    f: cp.Expression = 0
    for t in range(LinDistModelQ.n):
        if t in peak_h:
            peak = 0
            for a in "abc":
                peak += xk[model.idx("pij", model.SWING + 1, a, t)[0]]
            f += peak * peak_price * 10
        else:
            off_peak = 0
            for a in "abc":
                off_peak += xk[model.idx("pij", model.SWING + 1, a, t)[0]]
            f += off_peak * off_peak_price * 10
        for j in range(1, model.nb):
            for a in "abc":
                if model.phase_exists(a, t, j):
                    if LinDistModelQ.battery:
                        dis = model.idx("pd", j, a, t)
                        ch = model.idx("pc", j, a, t)
                        if ch:
                            f += 1e-3 * (1 - model.bat["nc_" + a].get(j, 1)) * (xk[ch])
                        if dis:
                            f += (
                                1e-3
                                * ((1 / model.bat["nd_" + a].get(j, 1)) - 1)
                                * (xk[dis])
                            )
                        # if dis:
                        #     f += 1e-3*((1/model.bat["nd_" + a].get(j,0))-model.bat["nc_" + a].get(j,0)) * (xk[dis])
    return f


def cp_obj_target_p_3ph(model, xk, **kwargs):
    f = cp.Constant(0)
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            f += (target[i] - p * (1 + loss_percent[i] / 100)) ** 2
    return f


def cp_obj_target_p_total(model, xk, **kwargs):
    actual = 0
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            actual += p
    f = (target - actual * (1 + loss_percent[0] / 100)) ** 2
    return f


def cp_obj_target_q_3ph(model, xk, **kwargs):
    target_q = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    f = cp.Constant(0)
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            f += (target_q[i] - q * (1 + loss_percent[i] / 100)) ** 2
    return f


def cp_obj_target_q_total(model, xk, **kwargs):
    actual = 0
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            actual += q
    f = (target - actual * (1 + loss_percent[0] / 100)) ** 2
    return f


def cp_obj_curtail(model, xk):
    f = cp.Constant(0)
    for i in range(model.p_der_start_phase_idx["a"], model.q_der_start_phase_idx["a"]):
        f += (model.bounds[i][1] - xk[i]) ** 2
    return f


def cp_obj_none(*args, **kwargs) -> cp.Constant:
    """
    For use with cvxpy_solve() to run a power flow with no optimization.

    Returns
    -------
    constant 0
    """
    return cp.Constant(0)


def cvxpy_solve(
    model: LinDistModel,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    """
    Solve a convex optimization problem using cvxpy.
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    obj_func : handle to the objective function
    kwargs :

    Returns
    -------
    result: scipy.optimize.OptimizeResult

    """
    m = model
    tic = perf_counter()
    solver = kwargs.get("solver", cp.OSQP)
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()
    x = cp.Variable(shape=(m.n_x,), name="x", value=x0)
    g = [csr_array(m.a_eq) @ x - m.b_eq.flatten() == 0]
    if model.battery:
        h = [csr_array(m.a_ineq) @ x - m.b_ineq.flatten() <= 0]
    else:
        h = []
    lb = [x[i] >= m.bounds[i][0] for i in range(m.n_x)]
    ub = [x[i] <= m.bounds[i][1] for i in range(m.n_x)]
    error_percent = kwargs.get("error_percent", np.zeros(3))
    target = kwargs.get("target", None)
    expression = obj_func(m, x, target=target, error_percent=error_percent)
    prob = cp.Problem(cp.Minimize(expression), g + h + ub + lb)
    prob.solve(verbose=False, solver=solver)

    x_res = x.value
    result = OptimizeResult(
        fun=prob.value,
        success=(prob.status == "optimal"),
        message=prob.status,
        x=x_res,
        nit=prob.solver_stats.num_iters,
        runtime=perf_counter() - tic,
    )
    return result


def lp_solve(model: LinDistModel, c: np.ndarray = None) -> OptimizeResult:
    """
    Solve a linear program using scipy.optimize.linprog and having the objective function:
        Min c^T x
    Parameters
    ----------
    model : LinDistModel
    c :  1-D array
        The coefficients of the linear objective function to be minimized.
    Returns
    -------
    result : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields
        below. Note that the return types of the fields may depend on whether
        the optimization was successful, therefore it is recommended to check
        `OptimizeResult.status` before relying on the other fields:

        x : 1-D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1-D array
            The (nominally positive) values of the slack variables,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

        nit : int
            The total number of iterations performed in all phases.
        message : str
            A string descriptor of the exit status of the algorithm.
    """
    if c is None:
        c = np.zeros(model.n_x)
    tic = perf_counter()
    if model.battery:
        res = linprog(
            c,
            A_eq=csr_array(model.a_eq),
            b_eq=model.b_eq.flatten(),
            A_ub=csr_array(model.a_ineq),
            b_ub=model.b_ineq.flatten(),
            bounds=model.bounds,
        )
    else:
        res = linprog(
            c,
            A_eq=csr_array(model.a_eq),
            b_eq=model.b_eq.flatten(),
            bounds=model.bounds,
        )
    if not res.success:
        raise ValueError(res.message)
    runtime = perf_counter() - tic
    res["runtime"] = runtime
    return res
