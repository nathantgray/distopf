from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_array, lil_array, vstack
from distopf.base import LinDistBase
from distopf.utils import get


class LinDistModelCapMI(LinDistBase):
    """
    LinDistFlow Model with support for capacitor bank control.

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
        self.build()

    def initialize_variable_index_pointers(self):
        self.x_maps, self.n_x = self._variable_tables(self.branch)
        self.v_map, self.n_x = self._add_device_variables(self.n_x, self.all_buses)
        self.pg_map, self.n_x = self._add_device_variables(self.n_x, self.gen_buses)
        self.qg_map, self.n_x = self._add_device_variables(self.n_x, self.gen_buses)
        self.qc_map, self.n_x = self._add_device_variables(self.n_x, self.cap_buses)
        self.vx_map, self.n_x = self._add_device_variables(self.n_x, self.reg_buses)
        self.zc_map, self.n_x = self._add_device_variables(self.n_x, self.cap_buses)
        self.uc_map, self.n_x = self._add_device_variables(self.n_x, self.cap_buses)

    def additional_variable_idx(self, var, node_j, phase):
        """
        User added index function. Override this function to add custom variables. Return None if `var` is not found.
        Parameters
        ----------
        var : name of variable
        node_j : node index (0 based; bus.id - 1)
        phase : "a", "b", or "c"

        Returns
        -------
        ix : index or list of indices of variable within x-vector or None if `var` is not found.
        """
        if var in ["zc"]:
            return self.zc_map[phase].get(node_j, [])
        if var in ["uc"]:
            return self.uc_map[phase].get(node_j, [])
        return None

    def add_capacitor_model(self, a_eq: csr_array, b_eq, j, a) -> (csr_array, np.ndarray):
        qij = self.idx("qij", j, a)
        q_cap_nom = 0
        if self.cap is not None:
            q_cap_nom = get(self.cap[f"q{a}"], j, 0)
        # equation indexes
        zc = self.idx("zc", j, a)
        qc = self.idx("q_cap", j, a)
        a_eq[qij, qc] = 1  # add capacitor q variable to power flow equation
        a_eq[qc, qc] = 1
        a_eq[qc, zc] = -q_cap_nom
        return a_eq, b_eq

    def create_capacitor_constraints(self) -> (csr_array, np.ndarray):
        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        n_rows_ineq = 4 * (
            len(self.cap_buses["a"])
            + len(self.cap_buses["b"])
            + len(self.cap_buses["c"])
        )
        a_ineq = lil_array((n_rows_ineq, self.n_x))
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
                a_ineq[ineq1, self.idx("zc", j, a)] = 1
                a_ineq[ineq1, self.idx("uc", j, a)] = -v_max
                a_ineq[ineq2, self.idx("zc", j, a)] = 1
                a_ineq[ineq2, self.idx("v", j, a)] = -1
                a_ineq[ineq3, self.idx("zc", j, a)] = -1
                a_ineq[ineq3, self.idx("v", j, a)] = +1
                a_ineq[ineq3, self.idx("uc", j, a)] = v_max
                b_ineq[ineq3] = v_max
                a_ineq[ineq4, self.idx("zc", j, a)] = -1
                ineq1 += 4
                ineq2 += 4
                ineq3 += 4
                ineq4 += 4

        return csr_array(a_ineq), b_ineq

    def create_inequality_constraints(self) -> (csr_array, np.ndarray):
        a_cap, b_cap = self.create_capacitor_constraints()
        a_inv, b_inv = self.create_octagon_constraints()
        a_ub = vstack([a_cap, a_inv])
        b_ub = np.r_[b_cap, b_inv]
        return csr_array(a_ub), b_ub

    def get_zc(self, x):
        return self.get_device_variables(x, self.zc_map)

    def get_uc(self, x):
        return self.get_device_variables(x, self.uc_map)
