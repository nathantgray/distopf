from collections.abc import Callable, Collection
from time import perf_counter
import distopf as opf
import numpy as np
import cvxpy as cp
import pandas as pd
import pyomo.environ as pe
from scipy.optimize import OptimizeResult
from distopf import (
    LinDistModel
)
from distopf.base import LinDistBase
from distopf.opf_solver import lp_solve


def cp_obj_loss(model: LinDistBase, x, **kwargs):
    """

    Parameters
    ----------
    model : LinDistModel
    xk : cp.Variable
    kwargs :

    Returns
    -------
    cost

    """
    index_list = []
    r_list = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        i = model.x_maps[a].bi
        j = model.x_maps[a].bj
        r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
        r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
        index_list = np.append(index_list, model.x_maps[a].pij.to_numpy().flatten())
        index_list = np.append(index_list, model.x_maps[a].qij.to_numpy().flatten())
    r = np.array(r_list)
    ix = np.array(index_list).astype(int)
    terms = []
    for i in range(len(ix)):
        terms.append(r*x[ix[i]**2])
    return sum(terms)


def pyo_obj_curtail(model: LinDistBase, x: pe.Var, **kwargs):
    """
    Objective function to minimize curtailment of DERs.
    Min sum((P_der_max - P_der)^2)
    Parameters
    ----------
    model : LinDistBase
    x : pe.Var

    Returns
    -------
    cost
    """

    all_pg_idx = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        all_pg_idx = np.r_[all_pg_idx, model.pg_map[a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    terms = []
    for i in range(len(all_pg_idx)):
        terms.append((model.x_max[all_pg_idx[i]] - x[all_pg_idx[i]]) ** 2)
    return sum(terms)


def pyomo_solve(
    model: LinDistBase,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    import pyomo.environ as pe

    m = model
    tic = perf_counter()
    solver = kwargs.get("solver", "ipopt")
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()

    cm = pe.ConcreteModel()
    cm.n_xk = pe.RangeSet(0, model.n_x - 1)
    cm.xk = pe.Var(cm.n_xk, initialize=x0)
    cm.constraints = pe.ConstraintList()
    for i in range(model.n_x):
        cm.constraints.add(cm.xk[i] <= model.x_max[i])
        cm.constraints.add(cm.xk[i] >= model.x_min[i])

    def equality_rule(_cm, i):
        if model.a_eq[[i], :].nnz > 0:
            return model.b_eq[i] == sum(
                _cm.xk[j] * model.a_eq[i, j]
                for j in range(model.n_x)
                if model.a_eq[i, j]
            )
        return pe.Constraint.Skip

    def inequality_rule(_cm, i):
        if model.a_ub[[i], :].nnz > 0:
            return model.b_ub[i] >= sum(
                _cm.xk[j] * model.a_ub[i, j]
                for j in range(model.n_x)
                if model.a_ub[i, j]
            )
        return pe.Constraint.Skip

    cm.equality = pe.Constraint(cm.n_xk, rule=equality_rule)
    if model.a_ub.shape[0] != 0:
        cm.ineq_set = pe.RangeSet(0, model.a_ub.shape[0] - 1)
        cm.inequality = pe.Constraint(cm.ineq_set, rule=inequality_rule)
    cm.objective = pe.Objective(expr=obj_func(model, cm.xk, **kwargs))
    opt = pe.SolverFactory(solver)
    results = opt.solve(cm)

    x_dict = cm.xk.extract_values()
    x_res = np.zeros(len(x_dict))
    for key, value in x_dict.items():
        x_res[key] = value

    result = OptimizeResult(
        fun=float(pe.value(cm.objective)),
        # success=(prob.status == "optimal"),
        # message=prob.status,
        x=x_res,
        # nit=prob.solver_stats.num_iters,
        runtime=perf_counter() - tic,
    )
    return result


def test():
    branch_data = pd.read_csv(opf.CASES_DIR / "csv" / "ieee123_30der/branch_data.csv")
    bus_data = pd.read_csv(opf.CASES_DIR / "csv" / "ieee123_30der/bus_data.csv")
    gen_data = pd.read_csv(opf.CASES_DIR / "csv" / "ieee123_30der/gen_data.csv")
    cap_data = pd.read_csv(opf.CASES_DIR / "csv" / "ieee123_30der/cap_data.csv")
    reg_data = pd.read_csv(opf.CASES_DIR / "csv" / "ieee123_30der/reg_data.csv")
    bus_data.loc[bus_data.bus_type == "SWING", ["v_a", "v_b", "v_c"]] = 1.011
    gen_data.control_variable = "P"
    model = LinDistModel(
        branch_data=branch_data,
        bus_data=bus_data,
        gen_data=gen_data,
        cap_data=cap_data,
        reg_data=reg_data,
    )
    result = pyomo_solve(model, pyo_obj_curtail)
    v = model.get_voltages(result.x)
    pg = model.get_p_gens(result.x)
    qg = model.get_q_gens(result.x)
    # s = model.get_apparent_power_flows(result.x)
    opf.plot_voltages(v).show()
    opf.plot_gens(pg, qg).show()


if __name__ == "__main__":
    test()
