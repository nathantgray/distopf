from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_matrix
from distopf.lindist_mi_base import LinDistModel, get


# bus_type options
SWING_FREE = "IN"
PQ_FREE = "OUT"
SWING_BUS = "SWING"
PQ_BUS = "PQ"


class LinDistModelQ(LinDistModel):
    """
    LinDistFlow Model with DER Reactive Power Injection as control variables.

    Parameters
    ----------
    branch_data : pd.DataFrame
        DataFrame containing branch data (r and x values, limits)
    bus_data : pd.DataFrame
        DataFrame containing bus data (loads, voltages, limits)
    gen_data : pd.DataFrame
        DataFrame containing generator/DER data
    cap_data : pd.DataFrame
        DataFrame containing capacitor data
    reg_data : pd.DataFrame
        DataFrame containing regulator data

    """

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


    def init_bounds(self, bus, gen):
        default = 100e3  # default value for variables that are not bounded.
        x_maps = self.x_maps
        # ~~~~~~~~~~ x limits ~~~~~~~~~~
        x_lim_lower = np.ones(self.n_x) * -default
        x_lim_upper = np.ones(self.n_x) * default
        for ph in "abc":
            if self.phase_exists(ph):
                x_lim_lower[x_maps[ph].loc[:, "pij"]] = -default  # P
                x_lim_upper[x_maps[ph].loc[:, "pij"]] = default  # P
                x_lim_lower[x_maps[ph].loc[:, "qij"]] = -default  # Q
                x_lim_upper[x_maps[ph].loc[:, "qij"]] = default  # Q
                # ~~ v limits ~~:
                i_root = list(set(x_maps["a"].bi) - set(x_maps["a"].bj))[0]
                i_v_swing = (
                    x_maps[ph]
                    .loc[x_maps[ph].loc[:, "bi"] == i_root, "vi"]
                    .to_numpy()[0]
                )
                x_lim_lower[i_v_swing] = bus.loc[i_root, "v_min"] ** 2
                x_lim_upper[i_v_swing] = bus.loc[i_root, "v_max"] ** 2
                x_lim_lower[x_maps[ph].loc[:, "vj"]] = (
                    bus.loc[x_maps[ph].loc[:, "bj"], "v_min"] ** 2
                )
                x_lim_upper[x_maps[ph].loc[:, "vj"]] = (
                    bus.loc[x_maps[ph].loc[:, "bj"], "v_max"] ** 2
                )
                # ~~ DER limits  ~~:
                for i in range(self.der_buses[ph].shape[0]):
                    i_q = self.qg_start_phase_idxs[ph] + i
                    # reactive power bounds
                    s_rated: pd.Series = gen[f"s{ph}_max"]
                    p_out: pd.Series = gen[f"p{ph}"]
                    q_min: pd.Series = -(((s_rated**2) - (p_out**2)) ** (1 / 2))
                    q_max: pd.Series = ((s_rated**2) - (p_out**2)) ** (1 / 2)
                    x_lim_lower[i_q] = q_min[self.der_buses[ph][i]]
                    x_lim_upper[i_q] = q_max[self.der_buses[ph][i]]
        bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
        return bounds

    def create_inequality_constraints(self):
        # ########## Aineq and Bineq Formation ###########
        n_rows_ineq = 4*3*self.cap.shape[0]
        a_ineq = zeros((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        for j in range(1, self.nb):
            def col(var, phase):
                return self.idx(var, j, phase)

            def row(var, phase):
                return self.idx(var, j, phase)

            for ph in ["abc", "bca", "cab"]:
                a = ph[0]
                if not self.phase_exists(a, j):
                    continue
                q_cap_nom = 0
                if self.cap is not None:
                    q_cap_nom = get(self.cap[f"q{a}"], j, 0)
                # equation indexes
                q_eqn = row("qij", a)

                a_eq[q_eqn, col("z_c", a)] = q_cap_nom

                v_max = get(self.bus["v_max"], j)**2
                a_ineq[ineq1, col("z_c", a)] = 1
                a_ineq[ineq1, col("u_c", a)] = -v_max
                a_ineq[ineq2, col("z_c", a)] = 1
                a_ineq[ineq2, col("vj", a)] = -1
                a_ineq[ineq3, col("z_c", a)] = -1
                a_ineq[ineq3, col("vj", a)] = +1
                a_ineq[ineq3, col("u_c", a)] = v_max
                b_ineq[ineq3] = v_max
                a_ineq[ineq4, col("z_c", a)] = -1


        return a_ineq, b_ineq
