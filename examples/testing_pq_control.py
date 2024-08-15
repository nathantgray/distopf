import numpy as np
from scipy.optimize import OptimizeResult

import distopf as opf
import pandas as pd
import cvxpy as cp
from time import perf_counter
def pq_solve(
    model: opf.LinDistPQ,
    obj_func,
    **kwargs,
):
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
    solver = kwargs.get("solver", cp.CLARABEL)
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = opf.lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()
    x = cp.Variable(shape=(m.n_x,), name="x", value=x0)
    g = [m.a_eq @ x - m.b_eq.flatten() == 0]
    lb = [x[i] >= m.bounds[i][0] for i in range(m.n_x)]
    ub = [x[i] <= m.bounds[i][1] for i in range(m.n_x)]
    g_inv = [m.a_ub @ x - m.b_ub <= 0]
    error_percent = kwargs.get("error_percent", np.zeros(3))
    target = kwargs.get("target", None)
    expression = obj_func(m, x, target=target, error_percent=error_percent)
    prob = cp.Problem(cp.Minimize(expression), g + g_inv + ub + lb)
    prob.solve(verbose=True, solver=solver)

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

if __name__ == '__main__':

    case = opf.DistOPFCase(
        data_path="ieee123_30der", gen_mult=3, load_mult=1, v_swing=1.0, v_max=1.05, v_min=0.95
    )
    # reg_data = pd.concat([case.reg_data, pd.DataFrame({
    # "fb": [128], "tb": [127], "phases": ["abc"], "tap_a": [15.0], "tap_b": [2.0], "tap_c": [5.0]})])
    case.cap_data = pd.concat([case.cap_data, pd.DataFrame({"id": [14],"name": [632],"qa": [0.3],"qb": [0.3],"qc": [0.5],"phases": ["abc"]})])
    case.gen_data.a_mode = "CONTROL_PQ"
    case.gen_data.b_mode = "CONTROL_PQ"
    case.gen_data.c_mode = "CONTROL_PQ"
    # case.gen_data.sa_max = case.gen_data.sa_max/1.2
    # case.gen_data.sb_max = case.gen_data.sb_max/1.2
    # case.gen_data.sc_max = case.gen_data.sc_max/1.2
    m = opf.LinDistPQ(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data
    )
    result = pq_solve(m, opf.cp_obj_loss)

    v = m.get_voltages(result.x)
    s = m.get_apparent_power_flows(result.x)
    pg = m.get_p_gens(result.x)
    qg = m.get_q_gens(result.x)
    dec_var = m.get_q_gens(result.x)
    print(result.fun)
    print(result.runtime)
    opf.plot_network(m, v, s, dec_var, "Q").show(renderer="browser")
    opf.plot_voltages(v).show(renderer="browser")
    opf.plot_power_flows(s).show(renderer="browser")
    opf.plot_ders(pg).show(renderer="browser")
    opf.plot_ders(qg).show(renderer="browser")
    opf.plot.plot_polar(pg, qg).show(renderer="browser")