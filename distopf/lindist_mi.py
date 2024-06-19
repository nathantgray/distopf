from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_matrix
from distopf.lindist_base_modular import LinDistModel, get


# bus_type options
SWING_FREE = "IN"
PQ_FREE = "OUT"
SWING_BUS = "SWING"
PQ_BUS = "PQ"


class LinDistModelMI(LinDistModel):
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
        self.a_ineq, self.b_ineq = self.create_inequality_constraints()


    def add_capacitor_model(self, a_eq, b_eq, j, phase):
        q_cap_nom = 0
        if self.cap is not None:
            q_cap_nom = get(self.cap[f"q{phase}"], j, 0)
        # equation indexes
        vj = self.idx("vj", j, phase)
        qc = self.idx("q_cap", j, phase)
        a_eq[qc, qc] = 1
        a_eq[qc, self.idx("z_c", j, phase)] = -q_cap_nom
        return a_eq, b_eq

    def create_inequality_constraints(self):
        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        n_rows_ineq = 4*(len(self.cap_buses["a"]) + len(self.cap_buses["b"]) + len(self.cap_buses["c"]))
        a_ineq = zeros((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq1 = 0
        ineq2 = 1
        ineq3 = 2
        ineq4 = 3
        for j in self.cap.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue
                # equation indexes
                v_max = get(self.bus["v_max"], j) ** 2
                a_ineq[ineq1, self.idx("z_c", j, a)] = 1
                a_ineq[ineq1, self.idx("u_c", j, a)] = -v_max
                a_ineq[ineq2, self.idx("z_c", j, a)] = 1
                a_ineq[ineq2, self.idx("vj", j, a)] = -1
                a_ineq[ineq3, self.idx("z_c", j, a)] = -1
                a_ineq[ineq3, self.idx("vj", j, a)] = +1
                a_ineq[ineq3, self.idx("u_c", j, a)] = v_max
                b_ineq[ineq3] = v_max
                a_ineq[ineq4, self.idx("z_c", j, a)] = -1
                ineq1 += 4
                ineq2 += 4
                ineq3 += 4
                ineq4 += 4

        return a_ineq, b_ineq


    def get_cap_statuses(self, x):
        nc_a = len(self.cap_buses["a"])
        nc_b = len(self.cap_buses["b"])
        nc_c = len(self.cap_buses["c"])
        nc = dict(a=nc_a, b=nc_b, c=nc_c)
        i_start = self.uc_start_phase_idxs
        cap_statuses = pd.DataFrame(columns=["name", "a", "b", "c"])
        if self.cap_data.shape[0] == 1 and self.cap_data.index[0] == -1:
            return cap_statuses
        for ph in "abc":
            for i_cap in range(nc[ph]):
                i = self.cap_buses[ph][i_cap]
                cap_statuses.at[i + 1, "name"] = self.bus.at[i, "name"]
                cap_statuses.at[i + 1, ph] = x[i_start[ph] + i_cap]
        return cap_statuses

    def get_cap_q(self, x):
        nc_a = len(self.cap_buses["a"])
        nc_b = len(self.cap_buses["b"])
        nc_c = len(self.cap_buses["c"])
        nc = dict(a=nc_a, b=nc_b, c=nc_c)
        i_start = self.qc_start_phase_idxs
        cap_q = pd.DataFrame(columns=["name", "a", "b", "c"])
        if self.cap_data.shape[0] == 1 and self.cap_data.index[0] == -1:
            return cap_q
        for ph in "abc":
            for i_cap in range(nc[ph]):
                i = self.cap_buses[ph][i_cap]
                cap_q.at[i + 1, "name"] = self.bus.at[i, "name"]
                cap_q.at[i + 1, ph] = x[i_start[ph] + i_cap]
        return cap_q
    
    def get_z_c(self, x):
        nc_a = len(self.cap_buses["a"])
        nc_b = len(self.cap_buses["b"])
        nc_c = len(self.cap_buses["c"])
        nc = dict(a=nc_a, b=nc_b, c=nc_c)
        i_start = self.zc_start_phase_idxs
        z_c = pd.DataFrame(columns=["name", "a", "b", "c"])
        if self.cap_data.shape[0] == 1 and self.cap_data.index[0] == -1:
            return z_c
        for ph in "abc":
            for i_zc in range(nc[ph]):
                i = self.cap_buses[ph][i_zc]
                z_c.at[i + 1, "name"] = self.bus.at[i, "name"]
                z_c.at[i + 1, ph] = x[i_start[ph] + i_zc]
        return z_c