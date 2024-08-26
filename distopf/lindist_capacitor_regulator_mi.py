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


class LinDistModelCapacitorRegulatorMI(opf.LinDistModelCapMI):
    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
    ):
        super().__init__(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            cap_data=cap_data,
            reg_data=reg_data,
        )
        self.x0 = None
        self.xk = None
        self.u_reg = None
        self.b_i = np.arange(0.9, 1.1, 0.00625)
        self.g_reg = None

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

    @classmethod
    def calculate_x0(cls, branch_data, bus_data, gen_data, cap_data, reg_data):
        bus_data = bus_data.copy()
        bus_data.loc[:, "v_min"] = 0.0
        bus_data.loc[:, "v_max"] = 2.0
        gen_data = gen_data.copy()
        gen_data.a_mode = "CONSTANT_PQ"
        gen_data.b_mode = "CONSTANT_PQ"
        gen_data.c_mode = "CONSTANT_PQ"
        m0 = cls(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            cap_data=cap_data,
            reg_data=reg_data,
        )
        result0 = opf.lp_solve(m0, np.zeros(m0.n_x))
        x0 = result0.x
        return x0

    def solve(self, obj_func: Callable, **kwargs) -> OptimizeResult:
        self.x0 = self.calculate_x0(
            self.branch_data, self.bus_data, self.gen_data, self.cap_data, self.reg_data
        )
        self.xk = cp.Variable(shape=(self.n_x,), name="x", value=self.x0)
        self.g_reg = self.cvxpy_regulator_mi_constraints()
        tic = perf_counter()
        solver = kwargs.get("solver", cp.SCIP)

        g = [csr_array(self.a_eq) @ self.xk - self.b_eq.flatten() == 0]
        g += [self.x_max >= self.xk, self.x_min <= self.xk]
        if self.a_ub is not None and self.b_ub is not None:
            if self.a_ub.shape[0] > 0:
                g += [self.a_ub @ self.xk - self.b_ub <= 0]
        error_percent = kwargs.get("error_percent", np.zeros(3))
        target = kwargs.get("target", None)
        expression = obj_func(self, self.xk, target=target, error_percent=error_percent)
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


# if __name__ == "__main__":
#
#     case = opf.DistOPFCase(
#         data_path="ieee123_30der",
#         gen_mult=20,
#         load_mult=1,
#         v_swing=1.0,
#         v_max=1.05,
#         v_min=0.95,
#     )
#     # reg_data = pd.concat(
#     #     [
#     #         case.reg_data,
#     #         pd.DataFrame(
#     #             {
#     #                 "fb": [128],
#     #                 "tb": [127],
#     #                 "phases": ["abc"],
#     #                 "tap_a": [15.0],
#     #                 "tap_b": [2.0],
#     #                 "tap_c": [5.0],
#     #             }
#     #         ),
#     #     ]
#     # )
#     # case.cap_data = pd.concat(
#     #     [
#     #         case.cap_data,
#     #         pd.DataFrame(
#     #             {
#     #                 "id": [14],
#     #                 "name": [632],
#     #                 "qa": [0.3],
#     #                 "qb": [0.3],
#     #                 "qc": [0.5],
#     #                 "phases": ["abc"],
#     #             }
#     #         ),
#     #     ]
#     # )
#     case.gen_data.a_mode = "CONTROL_PQ"
#     case.gen_data.b_mode = "CONTROL_PQ"
#     case.gen_data.c_mode = "CONTROL_PQ"
#     case.gen_data.sa_max = case.gen_data.sa_max / 1.2
#     case.gen_data.sb_max = case.gen_data.sb_max / 1.2
#     case.gen_data.sc_max = case.gen_data.sc_max / 1.2
#
#     m = LinDistModelCapacitorRegulatorMI(
#         branch_data=case.branch_data,
#         bus_data=case.bus_data,
#         gen_data=case.gen_data,
#         cap_data=case.cap_data,
#         reg_data=case.reg_data,
#     )
#     result = m.solve(opf.cp_obj_loss)
#
#     print(f" objective={result.fun}, time={result.runtime}")
#     print(m.u_reg.value @ np.array([m.b_i]).T)
#     v = m.get_voltages(result.x)
#     s = m.get_apparent_power_flows(result.x)
#     pg = m.get_p_gens(result.x)
#     qg = m.get_q_gens(result.x)
#     opf.plot_network(m, v, s, qg, "Q").show(renderer="browser")
#     opf.plot_voltages(v).show(renderer="browser")
#     opf.plot_power_flows(s).show(renderer="browser")
#     opf.plot_ders(pg).show(renderer="browser")
#     opf.plot_ders(qg).show(renderer="browser")
#     opf.plot.plot_polar(pg, qg).show(renderer="browser")
