import numpy as np
import distopf as opf

from collections.abc import Callable, Collection
from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array


import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_matrix
from distopf.lindist_base_modular import LinDistModelModular, get

def cvxpy_mi_reg_solve(
    model,
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
    solver = kwargs.get("solver", cp.SCIP)
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = opf.lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()
    x = cp.Variable(shape=(m.n_x,), name="x", value=x0)
    g = [csr_array(m.a_eq) @ x - m.b_eq.flatten() == 0]
    g_reg = m.create_algebraic_regulator_constraints(x)
    lb = [x >= m.x_min]
    ub = [x <= m.x_max]

    error_percent = kwargs.get("error_percent", np.zeros(3))
    target = kwargs.get("target", None)
    expression = obj_func(m, x, target=target, error_percent=error_percent)
    prob = cp.Problem(cp.Minimize(expression), g + ub + lb)
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


class MiReg(opf.LinDistPQ):
    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
    ):
        super().__init__(
            branch_data=branch_data, bus_data=bus_data, gen_data=gen_data, cap_data=cap_data, reg_data=reg_data
        )
        n_u_reg = len(self.reg_buses["a"]) + len(self.reg_buses["b"]) + len(self.reg_buses["c"])
        default_tap = np.zeros((max(n_u_reg, 1), 33))
        default_tap[:, 17] = 1
        self.u_reg = cp.Variable(shape=(max(n_u_reg, 1), 33), name="u_reg", value=default_tap, boolean=True)
        self.all_tap_ratios = np.arange(0.9, 1.1, 0.00625)


    def add_voltage_drop_model(self, a_eq, b_eq, j, a, b, c):
        r, x = self.r, self.x
        aa = "".join(sorted(a + a))
        # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
        ab = "".join(sorted(a + b))
        ac = "".join(sorted(a + c))
        i = self.idx("bi", j, a)[0]  # get the upstream node, i, on branch from i to j
        reg_ratio = 1
        if self.reg is not None:
            reg_ratio = get(self.reg[f"ratio_{a}"], j, 1)
        pij = self.idx("pij", j, a)
        qij = self.idx("qij", j, a)
        pijb = self.idx("pij", j, b)
        qijb = self.idx("qij", j, b)
        pijc = self.idx("pij", j, c)
        qijc = self.idx("qij", j, c)
        vi = self.idx("v", i, a)
        vj = self.idx("v", j, a)
        # Set V equation variable coefficients in a_eq and constants in b_eq
        if self.bus.bus_type[i] == opf.SWING_BUS:  # Swing bus
            a_eq[vi, vi] = 1
            b_eq[vi] = self.bus.at[i, f"v_{a}"] ** 2

        if self.reg is not None:
            if j in self.reg.tb:
                # reg_ratio = self.reg.at[j, f"ratio_{a}"]
                # a_eq[vj, vj] = 1
                # a_eq[vj, vi] = -1 * reg_ratio ** 2
                return a_eq, b_eq

        a_eq[vj, vj] = 1
        a_eq[vj, vi] = -1 * reg_ratio**2
        a_eq[vj, pij] = 2 * r[aa][i, j]
        a_eq[vj, qij] = 2 * x[aa][i, j]
        if self.phase_exists(b, j):
            a_eq[vj, pijb] = -r[ab][i, j] + sqrt(3) * x[ab][i, j]
            a_eq[vj, qijb] = -x[ab][i, j] - sqrt(3) * r[ab][i, j]
        if self.phase_exists(c, j):
            a_eq[vj, pijc] = -r[ac][i, j] - sqrt(3) * x[ac][i, j]
            a_eq[vj, qijc] = -x[ac][i, j] + sqrt(3) * r[ac][i, j]
        return a_eq, b_eq

    def create_algebraic_regulator_constraints(self, x):
        g_reg_sum = [cp.sum(self.u_reg, axis=1) == 1]
        g_reg = []
        big_m = 1e3
        i_reg = 0
        for j in self.reg.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue

                i = m.idx("bi", j, a)[0]
                vi = m.idx("v", i, a)
                vj = m.idx("v", j, a)
                for k in range(33):
                    g_reg = g_reg + [x[vj] - self.all_tap_ratios[k]**2*x[vi] - big_m*(1-self.u_reg[i_reg, k]) <= 0]
                    g_reg = g_reg + [x[vj] - self.all_tap_ratios[k]**2*x[vi] + big_m*(1-self.u_reg[i_reg, k]) >= 0]
                i_reg += 1
        return g_reg + g_reg_sum


if __name__ == '__main__':
    case = opf.DistOPFCase(
        data_path="ieee123", gen_mult=1, load_mult=1, v_swing=1.035, v_max=1.05, v_min=0.95
    )
    # case.reg_data = case.reg_data.iloc[:1, :]
    # case.bus_data.loc[case.bus_data.bus_type == opf.SWING_BUS, ["v_a", "v_b", "v_c"]] = 0.95
    bus_data = case.bus_data.copy()
    bus_data.loc[:, "v_min"] = 0.0
    bus_data.loc[:, "v_max"] = 2.0
    m0 = opf.LinDistModelModular(
        branch_data=case.branch_data,
        bus_data=bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data)
    result0 = opf.lp_solve(m0, np.zeros(m0.n_x))
    x0 = result0.x
    m = MiReg(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data
    )

    result = cvxpy_mi_reg_solve(m, opf.cp_obj_loss, x0=x0)

    # Solve model using provided objective function
    # result = opf.lp_solve(model, np.zeros(model.n_x))
    # # result = cvxpy_solve(model, cp_obj_loss)
    print(result.fun)
    print(m.u_reg.value@np.array([m.all_tap_ratios]).T)
    v = m.get_voltages(result.x)
    s = m.get_apparent_power_flows(result.x)
    dec_var = m.get_q_gens(result.x)
    opf.plot_network(m, v, s, dec_var, "Q").show(renderer="browser")
    opf.plot_voltages(v).show(renderer="browser")
    opf.plot_power_flows(s).show(renderer="browser")
