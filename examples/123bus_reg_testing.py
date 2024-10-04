import numpy as np
import distopf as opf

from collections.abc import Callable, Collection
from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array


import pandas as pd
from numpy import sqrt
from distopf.lindist_base_modular import get
from scipy.optimize import OptimizeResult


class LinDistModelRegulatorMI(opf.LinDistModelCapMI):
    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
    ):
        super().__init__(
            branch_data, bus_data, gen_data, cap_data=cap_data, reg_data=reg_data
        )
        self.x0 = self.calculate_x0()
        self.xk = cp.Variable(shape=(self.n_x,), name="x", value=self.x0)
        self.u_reg = None
        self.b_i = np.arange(0.9, 1.1, 0.00625)
        self.g_reg = self.cvxpy_regulator_mi_constraints()

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

    def create_inequality_constraints(self):
        a_cap, b_cap = self.create_capacitor_constraints()
        a_inv, b_inv = self.create_octagon_constraints()
        a_ub = np.r_[a_cap, a_inv]
        b_ub = np.r_[b_cap, b_inv]
        return a_ub, b_ub

    def cvxpy_regulator_mi_constraints(self):

        n_u_reg = (
            len(self.reg_buses["a"])
            + len(self.reg_buses["b"])
            + len(self.reg_buses["c"])
        )
        default_tap = np.zeros((max(n_u_reg, 1), 33))
        default_tap[:, 17] = 1
        self.u_reg = cp.Variable(
            shape=(max(n_u_reg, 1), 33), name="u_reg", value=default_tap, boolean=True
        )
        g_reg = [cp.sum(self.u_reg, axis=1) == 1]

        big_m = 1e3
        self.b_i = np.arange(0.9, 1.1, 0.00625)
        i_reg = 0
        for j in self.reg.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue

                i = self.idx("bi", j, a)[0]
                vi = self.idx("v", i, a)
                vj = self.idx("v", j, a)
                for k in range(33):
                    g_reg = g_reg + [
                        self.xk[vj]
                        - self.b_i[k] ** 2 * self.xk[vi]
                        - big_m * (1 - self.u_reg[i_reg, k])
                        <= 0
                    ]
                    g_reg = g_reg + [
                        self.xk[vj]
                        - self.b_i[k] ** 2 * self.xk[vi]
                        + big_m * (1 - self.u_reg[i_reg, k])
                        >= 0
                    ]
                i_reg += 1
        return g_reg

    def calculate_x0(self):
        bus_data = self.bus_data.copy()
        bus_data.loc[:, "v_min"] = 0.0
        bus_data.loc[:, "v_max"] = 2.0
        gen_data = self.gen_data.copy()
        gen_data.a_mode = "CONSTANT_PQ"
        gen_data.b_mode = "CONSTANT_PQ"
        gen_data.c_mode = "CONSTANT_PQ"
        m0 = opf.LinDistModelModular(
            branch_data=case.branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            cap_data=case.cap_data,
            reg_data=case.reg_data,
        )
        result0 = opf.lp_solve(m0, np.zeros(m0.n_x))
        x0 = result0.x
        return x0

    def solve(self, obj_func: Callable, **kwargs) -> OptimizeResult:
        tic = perf_counter()
        solver = kwargs.get("solver", cp.SCIP)

        g = [csr_array(self.a_eq) @ self.xk - self.b_eq.flatten() == 0]
        g += [self.x_max >= self.xk, self.x_min <= self.xk]
        if m.a_ub is not None and m.b_ub is not None:
            if m.a_ub.shape[0] > 0:
                g += [m.a_ub @ self.xk - m.b_ub <= 0]
        error_percent = kwargs.get("error_percent", np.zeros(3))
        target = kwargs.get("target", None)
        expression = obj_func(m, self.xk, target=target, error_percent=error_percent)
        prob = cp.Problem(cp.Minimize(expression), g + self.g_reg)
        prob.solve(solver=solver, verbose=False)

        x_res = self.xk.value
        result = OptimizeResult(
            fun=prob.value,
            success=(prob.status == "optimal"),
            message=prob.status,
            x=x_res,
            nit=prob.solver_stats.num_iters,
            runtime=perf_counter() - tic,
        )
        return result


if __name__ == "__main__":

    case = opf.DistOPFCase(
        data_path="ieee123",
        gen_mult=1,
        load_mult=1,
        v_swing=0.95,
        v_max=1.05,
        v_min=0.95,
    )

    m = LinDistModelRegulatorMI(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,
    )
    result = m.solve(opf.cp_obj_none)

    print(result.fun)
    print(m.u_reg.value @ np.array([m.b_i]).T)
    v = m.get_voltages(result.x)
    s = m.get_apparent_power_flows(result.x)
    dec_var = m.get_q_gens(result.x)
    opf.plot_network(m, v, s, dec_var, "Q").show(renderer="browser")
    opf.plot_voltages(v).show(renderer="browser")
    opf.plot_power_flows(s).show(renderer="browser")
