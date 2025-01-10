from time import perf_counter
import distopf as opf
import distopf.multiperiod as mpopf
from numpy import sqrt
import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array
import pyomo.environ as pe
from collections.abc import Callable, Collection
from scipy.optimize import OptimizeResult, linprog


def pyo_battery_efficiency(model: mpopf.LinDistModelMulti, xk: pe.Var | cp.Variable | np.ndarray, **kwargs):
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
    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    vec1_list = []
    index_list = []
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            charging_efficiency = model.bat.loc[model.charge_map[t][a].index, f"nc_{a}"].to_numpy()
            discharging_efficiency = model.bat.loc[model.discharge_map[t][a].index, f"nd_{a}"].to_numpy()
            vec1_list.extend((1 - charging_efficiency))
            vec1_list.extend(((1 / discharging_efficiency) - 1))
            index_list.extend(model.charge_map[t][a].to_numpy())
            index_list.extend(model.discharge_map[t][a].to_numpy())
    vec1 = np.array(vec1_list)
    ix = np.array(index_list)
    if isinstance(xk, pe.Var):
        return sum([vec1[i]*xk[ix[i]] for i in range(len(ix))])
    # if isinstance(xk, cp.Variable):
    #     return 1e-3 * cp.vdot(vec1, xk[ix])
    # else:
    #     return 1e-3 * np.vdot(vec1, xk[ix])


def pyo_voltage_reduction(model: opf.LinDistModel , x: pe.Var, **kwargs):
    v_list = []
    for a in "abc":
        if not model.phase_exists(a):
            continue
        for i in model.v_map[a]:
            v_list.append(x[i])
    return sum(v_list)


def pyo_voltage_reduction_mp(model: mpopf.LinDistModelMulti , x: pe.Var, **kwargs):
    v_list = []
    for t in range(model.start_step, model.start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a, t=t):
                continue
            for i in model.v_map[t][a]:
                v_list.append(x[i])
    return sum(v_list)


def pyo_storage_max(model: mpopf.LinDistModelMulti, x: pe.Var, **kwargs):
    b_list = []
    for t in range(model.start_step, model.start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a, t=t):
                continue
            for i in model.soc_map[t][a]:
                b_list.append(-x[i])
    return sum(b_list) + pyo_battery_efficiency(model, x, **kwargs)

def pyo_resiliency(model: mpopf.LinDistModelMulti, x: pe.Var, **kwargs):
    b_list = []
    for t in range(model.start_step, model.start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a, t=t):
                continue
            for ic, id in zip(model.charge_map[t][a], model.discharge_map[t][a]):
                p_batt = x[id] - x[ic]
                b_list.append(p_batt)
    score = sum(b_list)/np.sqrt(len(b_list))  ## why sqrt??
    return score


def pyo_decarb(model: mpopf.LinDistModelMulti, x: pe.Var, **kwargs):
    net_load = kwargs.get("net_load")
    gen_lists = {"a": [], "b": [], "c": []}
    for t in range(model.start_step, model.start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a, t=t):
                continue
            for ic, id, ipg in zip(model.charge_map[t][a], model.discharge_map[t][a], model.pg_map[t][a]):
                p_batt = x[id] - x[ic]
                gen_lists[a].append(p_batt)
                gen_lists[a].append(x[ipg])
    score = ((sum(gen_lists["a"]) - net_load[0])**2
             + (sum(gen_lists["b"]) - net_load[1])**2
             + (sum(gen_lists["c"]) - net_load[2])**2)
    return score


def pyo_flex(model: mpopf.LinDistModelMulti, x: pe.Var, **kwargs):
    rho = kwargs.get("rho")
    k = kwargs.get("k")
    if rho is None:
        raise ValueError("rho cannot be None")
    if k is None:
        raise ValueError("k cannot be None")
    score = 0
    if k > 1:
        score = -x[len(x) - 1] * (rho / 2)
    return score


def pyo_global(model: mpopf.LinDistModelMulti, x: pe.Var, **kwargs):
    cvr_rho = kwargs.pop("cvr_rho")
    decarb_rho = kwargs.pop("decarb_rho")
    res_rho = kwargs.pop("res_rho")
    return (pyo_resiliency(model, x, rho=res_rho, **kwargs)
            + pyo_flex(model, x, rho=res_rho, **kwargs)
            + pyo_decarb(model, x, rho=decarb_rho, **kwargs)
            + pyo_flex(model, x, rho=decarb_rho, **kwargs)
            + pyo_voltage_reduction_mp(model, x, rho=cvr_rho, **kwargs)
            + pyo_flex(model, x, rho=cvr_rho, **kwargs)
            + pyo_battery_efficiency(model, x, **kwargs)
            )


def pyo_cvr_rho(model: mpopf.LinDistModelMulti, x: pe.Var, **kwargs):
    cvr_rho = kwargs.pop("cvr_rho")
    return (pyo_voltage_reduction_mp(model, x, rho=cvr_rho, **kwargs)
            + pyo_flex(model, x, rho=cvr_rho, **kwargs)
            + pyo_battery_efficiency(model, x, **kwargs)
            )


def pyo_res_rho(model: mpopf.LinDistModelMulti, x: pe.Var, **kwargs):
    res_rho = kwargs.pop("res_rho")
    return (pyo_resiliency(model, x, rho=res_rho, **kwargs)
            + pyo_flex(model, x, rho=res_rho, **kwargs)
            + pyo_battery_efficiency(model, x, **kwargs)
            )


def pyo_decarb_rho(model: mpopf.LinDistModelMulti, x: pe.Var, **kwargs):
    decarb_rho = kwargs.pop("decarb_rho")
    return (pyo_resiliency(model, x, rho=decarb_rho, **kwargs)
            + pyo_flex(model, x, rho=decarb_rho, **kwargs)
            + pyo_battery_efficiency(model, x, **kwargs)
            )


def pyomo_solve(
    model: opf.LinDistModel | mpopf.LinDistModelMulti,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    import pyomo.environ as pe
    m = model
    tic = perf_counter()
    solver = kwargs.get("solver", "ipopt")
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = opf.lp_solve(m, np.zeros(m.n_x))
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
            return model.b_eq[i] == sum(_cm.xk[j]*model.a_eq[i, j] for j in range(model.n_x) if model.a_eq[i, j])
        return pe.Constraint.Skip
    def inequality_rule(_cm, i):
        if model.a_ub[[i], :].nnz > 0:
            return model.b_ub[i] >= sum(_cm.xk[j]*model.a_ub[i, j] for j in range(model.n_x) if model.a_ub[i, j])
        return pe.Constraint.Skip
    cm.equality = pe.Constraint(cm.n_xk, rule=equality_rule)
    if model.a_ub.shape[0] != 0:
        cm.ineq_set = pe.RangeSet(0, model.a_ub.shape[0] - 1)
        cm.inequality = pe.Constraint(cm.ineq_set, rule=inequality_rule)
    cm.objective = pe.Objective(expr=obj_func(model, cm.xk, **kwargs))
    pe.SolverFactory(solver).solve(cm)

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


def pyomo_flex_solve(
    model: opf.LinDistModel | mpopf.LinDistModelMulti,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    import pyomo.environ as pe
    m = model
    tic = perf_counter()
    solver = kwargs.get("solver", "ipopt")
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = opf.lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()

    cm = pe.ConcreteModel()
    cm.n_xk = pe.RangeSet(0, model.n_x - 1)
    cm.n_xk_flex = pe.RangeSet(0, model.n_x)  # last index is for flex variable
    cm.xk = pe.Var(cm.n_xk_flex, initialize=np.append(x0, [0]))
    cm.constraints = pe.ConstraintList()
    for i in range(model.n_x):
        cm.constraints.add(cm.xk[i] <= model.x_max[i])
        cm.constraints.add(cm.xk[i] >= model.x_min[i])

    def equality_rule(_cm, i):
        if model.a_eq[[i], :].nnz > 0:
            return model.b_eq[i] == sum(_cm.xk[j]*model.a_eq[i, j] for j in range(model.n_x) if model.a_eq[i, j])
        return pe.Constraint.Skip
    def inequality_rule(_cm, i):
        if model.a_ub[[i], :].nnz > 0:
            return model.b_ub[i] >= sum(_cm.xk[j]*model.a_ub[i, j] for j in range(model.n_x) if model.a_ub[i, j])
        return pe.Constraint.Skip
    cm.equality = pe.Constraint(cm.n_xk, rule=equality_rule)
    if model.a_ub.shape[0] != 0:
        cm.ineq_set = pe.RangeSet(0, model.a_ub.shape[0] - 1)
        cm.inequality = pe.Constraint(cm.ineq_set, rule=inequality_rule)
    cm.objective = pe.Objective(expr=obj_func(model, cm.xk, **kwargs))
    pe.SolverFactory(solver).solve(cm)

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
    branch_data = pd.read_csv("ieee13_battery/branch_data.csv")
    bus_data = pd.read_csv("ieee13_battery/bus_data.csv")
    gen_data = pd.read_csv("ieee13_battery/gen_data.csv")
    cap_data = pd.read_csv("ieee13_battery/cap_data.csv")
    reg_data = pd.read_csv("ieee13_battery/reg_data.csv")
    bat_data = pd.read_csv("ieee13_battery/battery_data.csv")
    load_profile = pd.read_csv("Profiles/ld_shape.csv", names=["M"])
    load_profile["time"] = load_profile.index
    pv_profile = pd.read_csv("Profiles/pv_shape.csv", names=["PV"])
    pv_profile["time"] = pv_profile.index
    bus_data.v_max = 1.2
    bus_data.v_min = 0.8
    bus_data.cvr_p = 2
    bus_data.cvr_q = 2
    gen_data.control_variable = "PQ"
    model = mpopf.LinDistModelMultiFast(
        branch_data=branch_data,
        bus_data=bus_data,
        gen_data=gen_data,
        cap_data=cap_data,
        reg_data=reg_data,
        bat_data=bat_data,
        loadshape_data=load_profile,
        pv_loadshape_data=pv_profile,
        n_steps=4,
        start_step=45,
        delta_t=0.25,  # 15 minutes


    )
    result = pyomo_solve(model, pyo_storage_max)
    v = model.get_voltages(result.x)
    v = v.loc[v.t==model.start_step, ["id", "name", "a", "b", "c"]]
    # s = model.get_apparent_power_flows(result.x)
    opf.plot_voltages(v).show()
    print(model.get_p_charge(result.x))
    print(model.get_p_discharge(result.x))
    print(model.get_soc(result.x))


if __name__ == '__main__':
    test()