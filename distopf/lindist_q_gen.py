from typing import Optional
from functools import cache
import numpy as np
import pandas as pd
import distopf as opf
from distopf.base import LinDistBase
from distopf.utils import get


class LinDistModelQGen(LinDistBase):
    """
    LinDistFlow Model with reactive power control of generators.
    No variables for active power generation exist. This can make the
    model smaller and run faster if there are many generators and only
    reactive power control is required.

    Parameters
    ----------
    branch_data : pd.DataFrame
        DataFrame containing branch data including resistance and reactance values and limits.
    bus_data : pd.DataFrame
        DataFrame containing bus data such as loads, voltages, and limits.
    gen_data : pd.DataFrame
        DataFrame containing generator data.
    cap_data : pd.DataFrame
        DataFrame containing capacitor data.
    reg_data : pd.DataFrame
        DataFrame containing regulator data.

    """

    def __init__(
        self,
        branch_data: Optional[pd.DataFrame] = None,
        bus_data: Optional[pd.DataFrame] = None,
        gen_data: Optional[pd.DataFrame] = None,
        cap_data: Optional[pd.DataFrame] = None,
        reg_data: Optional[pd.DataFrame] = None,
    ):
        super().__init__(branch_data, bus_data, gen_data, cap_data, reg_data)
        self.build()

    def initialize_variable_index_pointers(self):
        self.x_maps, self.n_x = self._variable_tables(self.branch)
        self.v_map, self.n_x = self._add_device_variables(self.n_x, self.all_buses)
        self.qg_map, self.n_x = self._add_device_variables(self.n_x, self.gen_buses)
        self.qc_map, self.n_x = self._add_device_variables(self.n_x, self.cap_buses)
        self.vx_map, self.n_x = self._add_device_variables(self.n_x, self.reg_buses)

    def add_generator_limits(self, x_lim_lower, x_lim_upper):
        for a in "abc":
            if not self.phase_exists(a):
                continue
            s_rated = self.gen[f"s{a}_max"]
            p_out = self.gen[f"p{a}"]
            q_max = ((s_rated**2) - (p_out**2)) ** (1 / 2)
            q_min = -q_max
            q_max_manual = self.gen.get(f"q{a}_max", np.ones_like(q_min) * 100e3)
            q_min_manual = self.gen.get(f"q{a}_min", np.ones_like(q_min) * -100e3)
            for j in self.gen_buses[a]:
                qg = self.idx("qg", j, a)
                # reactive power bounds
                x_lim_lower[qg] = max(q_min[j], q_min_manual[j])
                x_lim_upper[qg] = min(q_max[j], q_max_manual[j])
        return x_lim_lower, x_lim_upper

    @cache
    def idx(self, var, node_j, phase):
        if var in self.x_maps[phase].columns:
            return self.branch_into_j(var, node_j, phase)
        if var in ["pjk"]:  # indexes of all branch active power out of node j
            return self.branches_out_of_j("pij", node_j, phase)
        if var in ["qjk"]:  # indexes of all branch reactive power out of node j
            return self.branches_out_of_j("qij", node_j, phase)
        if var in ["v"]:  # active power generation at node
            return get(self.v_map[phase], node_j, [])
        if var in ["qg", "q_gen"]:  # reactive power generation at node
            return get(self.qg_map[phase], node_j, [])
        if var in ["qc", "q_cap"]:  # reactive power injection by capacitor
            return get(self.qc_map[phase], node_j, [])
        if var in ["vx"]:
            return self.vx_map[phase].get(node_j, [])
        ix = self.additional_variable_idx(var, node_j, phase)
        if ix is not None:
            return ix
        raise ValueError(f"Variable name, '{var}', not found.")

    def add_power_flow_model(self, a_eq, b_eq, j, phase):
        pij = self.idx("pij", j, phase)
        qij = self.idx("qij", j, phase)
        pjk = self.idx("pjk", j, phase)
        qjk = self.idx("qjk", j, phase)
        qg = self.idx("qg", j, phase)
        qc = self.idx("q_cap", j, phase)
        vj = self.idx("v", j, phase)
        p_gen_nom = 0, 0
        if self.gen is not None:
            p_gen_nom = get(self.gen[f"p{phase}"], j, 0)
        p_load_nom, q_load_nom = 0, 0
        if self.bus.bus_type[j] == opf.PQ_BUS:
            p_load_nom = self.bus[f"pl_{phase}"][j]
            q_load_nom = self.bus[f"ql_{phase}"][j]
        # Set P equation variable coefficients in a_eq
        a_eq[pij, pij] = 1
        a_eq[pij, pjk] = -1
        # Set Q equation variable coefficients in a_eq
        a_eq[qij, qij] = 1
        a_eq[qij, qjk] = -1
        a_eq[qij, qg] = 1
        a_eq[qij, qc] = 1
        if self.bus.bus_type[j] != opf.PQ_FREE:
            # Set Load equation variable coefficients in a_eq
            a_eq[pij, vj] = -(self.bus.cvr_p[j] / 2) * p_load_nom
            b_eq[pij] = (1 - (self.bus.cvr_p[j] / 2)) * p_load_nom - p_gen_nom
            a_eq[qij, vj] = -(self.bus.cvr_q[j] / 2) * q_load_nom
            b_eq[qij] = (1 - (self.bus.cvr_q[j] / 2)) * q_load_nom
        return a_eq, b_eq

    def add_generator_model(self, a_eq, b_eq, j, phase):
        return a_eq, b_eq

    def add_load_model(self, a_eq, b_eq, j, phase):
        return a_eq, b_eq

    def add_capacitor_model(self, a_eq, b_eq, j, phase):
        q_cap_nom = 0
        if self.cap is not None:
            q_cap_nom = get(self.cap[f"q{phase}"], j, 0)
        # equation indexes
        vj = self.idx("v", j, phase)
        qc = self.idx("q_cap", j, phase)
        a_eq[qc, qc] = 1
        a_eq[qc, vj] = -q_cap_nom
        return a_eq, b_eq

    def create_inequality_constraints(self):
        return None, None

    def get_p_gens(self, x):
        df = self.get_device_variables(x, self.qg_map)
        df.a = self.gen_data.pa.to_numpy()
        df.b = self.gen_data.pb.to_numpy()
        df.c = self.gen_data.pc.to_numpy()
        return df

    def get_apparent_power_flows(self, x):
        s_df = pd.DataFrame(
            columns=["fb", "tb", "from_name", "to_name", "a", "b", "c"],
            index=range(2, self.nb + 1),
        )
        s_df["a"] = s_df["a"].astype(complex)
        s_df["b"] = s_df["b"].astype(complex)
        s_df["c"] = s_df["c"].astype(complex)
        for ph in "abc":
            fb_idxs = self.x_maps[ph].bi.to_numpy()
            fb_names = self.bus.name[fb_idxs].to_numpy()
            tb_idxs = self.x_maps[ph].bj.to_numpy()
            tb_names = self.bus.name[tb_idxs].to_numpy()
            s_df.loc[self.x_maps[ph].bj.to_numpy() + 1, "fb"] = fb_idxs + 1
            s_df.loc[self.x_maps[ph].bj.to_numpy() + 1, "tb"] = tb_idxs + 1
            s_df.loc[self.x_maps[ph].bj.to_numpy() + 1, "from_name"] = fb_names
            s_df.loc[self.x_maps[ph].bj.to_numpy() + 1, "to_name"] = tb_names
            s_df.loc[self.x_maps[ph].bj.to_numpy() + 1, ph] = (
                x[self.x_maps[ph].pij] + 1j * x[self.x_maps[ph].qij]
            )
        return s_df
