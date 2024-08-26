import numpy as np
from scipy.optimize import OptimizeResult

import distopf as opf
import pandas as pd
import cvxpy as cp
from time import perf_counter


class Hexagon(opf.LinDistModelPQ):
    def create_inequality_constraints(self):
        return self.create_hexagon_constraints()


class Octogon(opf.LinDistModelPQ):
    def create_inequality_constraints(self):
        return self.create_octagon_constraints()


if __name__ == "__main__":

    case = opf.DistOPFCase(
        data_path="ieee123_30der",
        gen_mult=6,
        load_mult=1,
        v_swing=1.0,
        v_max=1.05,
        v_min=0.95,
    )
    reg_data = pd.concat(
        [
            case.reg_data,
            pd.DataFrame(
                {
                    "fb": [128],
                    "tb": [127],
                    "phases": ["abc"],
                    "tap_a": [15.0],
                    "tap_b": [2.0],
                    "tap_c": [5.0],
                }
            ),
        ]
    )
    case.cap_data = pd.concat(
        [
            case.cap_data,
            pd.DataFrame(
                {
                    "id": [14],
                    "name": [632],
                    "qa": [0.3],
                    "qb": [0.3],
                    "qc": [0.5],
                    "phases": ["abc"],
                }
            ),
        ]
    )
    case.gen_data.a_mode = "CONTROL_PQ"
    case.gen_data.b_mode = "CONTROL_PQ"
    case.gen_data.c_mode = "CONTROL_PQ"
    case.gen_data.sa_max = case.gen_data.sa_max / 1.2
    case.gen_data.sb_max = case.gen_data.sb_max / 1.2
    case.gen_data.sc_max = case.gen_data.sc_max / 1.2
    hex = Hexagon(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,
    )

    oct = Octogon(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,
    )

    result_hex = opf.cvxpy_solve(hex, opf.cp_obj_loss)
    print(f" Hexagon: objective={result_hex.fun}, time={result_hex.runtime}")
    # v = m.get_voltages(result.x)
    # s = m.get_apparent_power_flows(result.x)
    pg_hex = hex.get_p_gens(result_hex.x)
    qg_hex = hex.get_q_gens(result_hex.x)
    opf.plot.plot_polar(pg_hex, qg_hex).show(renderer="browser")

    result_oct = opf.cvxpy_solve(oct, opf.cp_obj_loss)
    print(f" Octogon: objective={result_oct.fun}, time={result_oct.runtime}")
    pg_oct = hex.get_p_gens(result_oct.x)
    qg_oct = hex.get_q_gens(result_oct.x)
    opf.plot.plot_polar(pg_oct, qg_oct).show(renderer="browser")

    # dec_var = m.get_q_gens(result.x)
    # print(result.fun)
    # print(result.runtime)
    # opf.plot_network(m, v, s, dec_var, "Q").show(renderer="browser")
    # opf.plot_voltages(v).show(renderer="browser")
    # opf.plot_power_flows(s).show(renderer="browser")
    # opf.plot_ders(pg).show(renderer="browser")
    # opf.plot_ders(qg).show(renderer="browser")
    # opf.plot.plot_polar(pg, qg).show(renderer="browser")
