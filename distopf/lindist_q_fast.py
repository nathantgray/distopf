from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_array
import distopf as opf


def get(s: pd.Series, i, default=None):
    """
    Get value at index i from a Series. Return default if it does not exist.
    Parameters
    ----------
    s : pd.Series
    i : index or key for eries
    default : value to return if it fails

    Returns
    -------
    value: value at index i or default if it doesn't exist.
    """
    try:
        return s.loc[i]
    except (KeyError, ValueError, IndexError):
        return default


def _handle_gen_input(gen_data: pd.DataFrame) -> pd.DataFrame:
    if gen_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "pa",
                "pb",
                "pc",
                "qa",
                "qb",
                "qc",
                "sa_max",
                "sb_max",
                "sc_max",
                "phases",
                "qa_max",
                "qb_max",
                "qc_max",
                "qa_min",
                "qb_min",
                "qc_min",
                "a_mode",
                "b_mode",
                "c_mode",
            ]
        )
    for ph in "abc":
        if f"{ph}_mode" not in gen_data.columns:
            gen_data[f"{ph}_mode"] = 0
    gen = gen_data.sort_values(by="id", ignore_index=True)
    gen.index = gen.id.to_numpy() - 1
    return gen


def _handle_cap_input(cap_data: pd.DataFrame) -> pd.DataFrame:
    if cap_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "qa",
                "qb",
                "qc",
                "phases",
            ]
        )
    cap = cap_data.sort_values(by="id", ignore_index=True)
    cap.index = cap.id.to_numpy() - 1
    return cap


def _handle_reg_input(reg_data: pd.DataFrame) -> pd.DataFrame:
    if reg_data is None:
        return pd.DataFrame(
            columns=[
                "fb",
                "tb",
                "phases",
                "tap_a",
                "tap_b",
                "tap_c",
                "ratio_a",
                "ratio_b",
                "ratio_c",
            ]
        )
    reg = reg_data.sort_values(by="tb", ignore_index=True)
    reg.index = reg.tb.to_numpy() - 1
    for ph in "abc":
        if f"tap_{ph}" in reg.columns and not f"ratio_{ph}" in reg.columns:
            reg[f"ratio_{ph}"] = 1 + 0.00625 * reg[f"tap_{ph}"]
        elif f"ratio_{ph}" in reg.columns and not f"tap_{ph}" in reg.columns:
            reg[f"tap_{ph}"] = (reg[f"ratio_{ph}"] - 1) / 0.00625
        elif f"ratio_{ph}" in reg.columns and f"tap_{ph}" in reg.columns:
            reg[f"ratio_{ph}"] = 1 + 0.00625 * reg[f"tap_{ph}"]
            # check consistency
            # if any(abs(reg[f"ratio_{ph}"]) - (1 + 0.00625 * reg[f"tap_{ph}"]) > 1e-6):
            #     raise ValueError(
            #         f"Regulator taps and ratio are inconsistent on phase {ph}!"
            #     )
    return reg


def _handle_branch_input(branch_data: pd.DataFrame) -> pd.DataFrame:
    if branch_data is None:
        raise ValueError("Branch data must be provided.")
    branch = branch_data.sort_values(by="tb", ignore_index=True)
    branch = branch.loc[branch.status != "OPEN", :]
    return branch


def _handle_bus_input(bus_data: pd.DataFrame) -> pd.DataFrame:
    if bus_data is None:
        raise ValueError("Bus data must be provided.")
    bus = bus_data.sort_values(by="id", ignore_index=True)
    bus.index = bus.id.to_numpy() - 1
    return bus


class LinDistModelQFast:
    """
    LinDistFlow Model base class.

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
        # ~~~~~~~~~~~~~~~~~~~~ Load Data Frames ~~~~~~~~~~~~~~~~~~~~
        self.branch = _handle_branch_input(branch_data)
        self.bus = _handle_bus_input(bus_data)
        self.gen = _handle_gen_input(gen_data)
        self.cap = _handle_cap_input(cap_data)
        self.reg = _handle_reg_input(reg_data)

        # ~~~~~~~~~~~~~~~~~~~~ prepare data ~~~~~~~~~~~~~~~~~~~~
        self.nb = len(self.bus.id)
        self.r, self.x = self._init_rx(self.branch)
        self.all_buses = {
            "a": self.bus.loc[self.bus.phases.str.contains("a")].index.to_numpy(),
            "b": self.bus.loc[self.bus.phases.str.contains("b")].index.to_numpy(),
            "c": self.bus.loc[self.bus.phases.str.contains("c")].index.to_numpy(),
        }
        self.gen_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.gen.shape[0] > 0:
            self.gen_buses = {
                "a": self.gen.loc[self.gen.phases.str.contains("a")].index.to_numpy(),
                "b": self.gen.loc[self.gen.phases.str.contains("b")].index.to_numpy(),
                "c": self.gen.loc[self.gen.phases.str.contains("c")].index.to_numpy(),
            }
            self.n_gens = (
                len(self.gen_buses["a"])
                + len(self.gen_buses["b"])
                + len(self.gen_buses["c"])
            )
        self.cap_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.cap.shape[0] > 0:
            self.cap_buses = {
                "a": self.cap.loc[self.cap.phases.str.contains("a")].index.to_numpy(),
                "b": self.cap.loc[self.cap.phases.str.contains("b")].index.to_numpy(),
                "c": self.cap.loc[self.cap.phases.str.contains("c")].index.to_numpy(),
            }
            self.n_caps = (
                len(self.cap_buses["a"])
                + len(self.cap_buses["b"])
                + len(self.cap_buses["c"])
            )
        self.reg_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.reg.shape[0] > 0:
            self.reg_buses = {
                "a": self.reg.loc[self.reg.phases.str.contains("a")].index.to_numpy(),
                "b": self.reg.loc[self.reg.phases.str.contains("b")].index.to_numpy(),
                "c": self.reg.loc[self.reg.phases.str.contains("c")].index.to_numpy(),
            }
            self.n_regs = (
                len(self.reg_buses["a"])
                + len(self.reg_buses["b"])
                + len(self.reg_buses["c"])
            )
        # ~~ initialize index pointers ~~
        self.x_maps, self.n_x = self._variable_tables(self.branch)
        self.v_map, self.n_x = self._add_device_variables(self.n_x, self.all_buses)
        self.qg_map, self.n_x = self._add_device_variables(self.n_x, self.gen_buses)
        self.qc_map, self.n_x = self._add_device_variables(self.n_x, self.cap_buses)
        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq ~~~~~~~~~~~~~~~~~~~~
        self.a_eq, self.b_eq = self.create_model()
        self.a_ub, self.b_ub = self.create_inequality_constraints()
        self._bounds = self.init_bounds()
        self.bounds = list(map(tuple, self._bounds))
        self.x_min = self._bounds[:, 0]
        self.x_max = self._bounds[:, 1]

    @staticmethod
    def _init_rx(branch):
        row = np.array(np.r_[branch.fb, branch.tb], dtype=int) - 1
        col = np.array(np.r_[branch.tb, branch.fb], dtype=int) - 1
        r = {
            "aa": csr_array((np.r_[branch.raa, branch.raa], (row, col))),
            "ab": csr_array((np.r_[branch.rab, branch.rab], (row, col))),
            "ac": csr_array((np.r_[branch.rac, branch.rac], (row, col))),
            "bb": csr_array((np.r_[branch.rbb, branch.rbb], (row, col))),
            "bc": csr_array((np.r_[branch.rbc, branch.rbc], (row, col))),
            "cc": csr_array((np.r_[branch.rcc, branch.rcc], (row, col))),
        }
        x = {
            "aa": csr_array((np.r_[branch.xaa, branch.xaa], (row, col))),
            "ab": csr_array((np.r_[branch.xab, branch.xab], (row, col))),
            "ac": csr_array((np.r_[branch.xac, branch.xac], (row, col))),
            "bb": csr_array((np.r_[branch.xbb, branch.xbb], (row, col))),
            "bc": csr_array((np.r_[branch.xbc, branch.xbc], (row, col))),
            "cc": csr_array((np.r_[branch.xcc, branch.xcc], (row, col))),
        }
        return r, x

    @staticmethod
    def _variable_tables(branch, n_x=0):
        x_maps = {}
        for a in "abc":
            indices = branch.phases.str.contains(a)
            lines = branch.loc[indices, ["fb", "tb"]].values.astype(int) - 1
            n_lines = len(lines)
            df = pd.DataFrame(columns=["bi", "bj", "pij", "qij"], index=range(n_lines))
            if n_lines == 0:
                x_maps[a] = df.astype(int)
                continue
            g = nx.Graph()
            g.add_edges_from(lines)
            i_root = list(set(lines[:, 0]) - set(lines[:, 1]))[
                0
            ]  # root node is only node with no from-bus
            edges = np.array(list(nx.dfs_edges(g, source=i_root)))
            df["bi"] = edges[:, 0]
            df["bj"] = edges[:, 1]
            df["pij"] = np.array([i for i in range(n_x, n_x + n_lines)])
            n_x = n_x + n_lines
            df["qij"] = np.array([i for i in range(n_x, n_x + n_lines)])
            n_x = n_x + n_lines
            x_maps[a] = df.astype(int)
        return x_maps, n_x

    @staticmethod
    def _add_device_variables(n_x: int, device_buses: dict):
        n_a = len(device_buses["a"])
        n_b = len(device_buses["b"])
        n_c = len(device_buses["c"])
        device_maps = {
            "a": pd.Series(range(n_x, n_x + n_a), index=device_buses["a"]),
            "b": pd.Series(range(n_x + n_a, n_x + n_a + n_b), index=device_buses["b"]),
            "c": pd.Series(
                range(n_x + n_a + n_b, n_x + n_a + n_b + n_c), index=device_buses["c"]
            ),
        }
        n_x = n_x + n_a + n_b + n_c
        return device_maps, n_x

    def init_bounds(self):
        default = 100e3  # Default for unbounded variables.
        # ~~~~~~~~~~ x limits ~~~~~~~~~~
        x_lim_lower = np.ones(self.n_x) * -default
        x_lim_upper = np.ones(self.n_x) * default
        x_lim_lower, x_lim_upper = self.add_voltage_limits(x_lim_lower, x_lim_upper)
        x_lim_lower, x_lim_upper = self.add_generator_limits(x_lim_lower, x_lim_upper)
        x_lim_lower, x_lim_upper = self.user_added_limits(x_lim_lower, x_lim_upper)
        bounds = np.c_[x_lim_lower, x_lim_upper]
        # bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
        return bounds

    def user_added_limits(self, x_lim_lower, x_lim_upper):
        """
        User added limits function. Override this function to add custom variable limits.
        Parameters
        ----------
        x_lim_lower :
        x_lim_upper :

        Returns
        -------
        x_lim_lower : lower limits for x-vector
        x_lim_upper : upper limits for x-vector

        Examples
        --------
        ```python
        p_lim = 10
        q_lim = 10
        for a in "abc":
            if not self.phase_exists(a):
                continue
            x_lim_lower[self.x_maps[a].pij] = -p_lim
            x_lim_upper[self.x_maps[a].pij] = p_lim
            x_lim_lower[self.x_maps[a].qij] = -q_lim
            x_lim_upper[self.x_maps[a].qij] = q_lim
        ```
        """
        return x_lim_lower, x_lim_upper

    def add_voltage_limits(self, x_lim_lower, x_lim_upper):
        for a in "abc":
            if not self.phase_exists(a):
                continue
            # ~~ v limits ~~:
            x_lim_upper[self.v_map[a]] = self.bus.loc[self.v_map[a].index, "v_max"] ** 2
            x_lim_lower[self.v_map[a]] = self.bus.loc[self.v_map[a].index, "v_min"] ** 2
        return x_lim_lower, x_lim_upper

    def add_generator_limits(self, x_lim_lower, x_lim_upper):
        for a in "abc":
            if not self.phase_exists(a):
                continue
            q_max_manual = self.gen.get(f"q{a}_max", 100e3)
            q_min_manual = self.gen.get(f"q{a}_min", -100e3)
            s_rated = self.gen[f"s{a}_max"]
            p_out = self.gen[f"p{a}"]
            q_max = ((s_rated**2) - (p_out**2)) ** (1 / 2)
            q_min = -1 * q_max
            for j in self.gen_buses[a]:
                qg = self.idx("qg", j, a)
                # reactive power bounds
                x_lim_lower[qg] = max(q_min[j], q_min_manual[j])
                x_lim_upper[qg] = min(q_max[j], q_max_manual[j])
        return x_lim_lower, x_lim_upper

    @cache
    def branch_into_j(self, var, j, phase):
        idx = self.x_maps[phase].loc[self.x_maps[phase].bj == j, var].to_numpy()
        return idx[~np.isnan(idx)].astype(int)

    @cache
    def branches_out_of_j(self, var, j, phase):
        idx = self.x_maps[phase].loc[self.x_maps[phase].bi == j, var].to_numpy()
        return idx[~np.isnan(idx)].astype(int)

    @cache
    def idx(self, var, node_j, phase):
        if var in self.x_maps[phase].columns:
            return self.branch_into_j(var, node_j, phase)
        if var in ["pjk"]:  # indexes of all branch active power out of node j
            return self.branches_out_of_j("pij", node_j, phase)
        if var in ["qjk"]:  # indexes of all branch reactive power out of node j
            return self.branches_out_of_j("qij", node_j, phase)
        if var in ["v"]:  # active power generation at node
            return self.v_map[phase].get(node_j, [])
        if var in ["qg", "q_gen"]:  # reactive power generation at node
            return self.qg_map[phase].get(node_j, [])
        if var in ["qc", "q_cap"]:  # reactive power injection by capacitor
            return self.qc_map[phase].get(node_j, [])
        ix = self.user_added_idx(var, node_j, phase)
        if ix is not None:
            return ix
        raise ValueError(f"Variable name, '{var}', not found.")

    def user_added_idx(self, var, node_j, phase):
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
        return None

    @cache
    def phase_exists(self, phase, index: int = None):
        if index is None:
            return self.x_maps[phase].shape[0] > 0
        return len(self.idx("bj", index, phase)) > 0

    def create_model(self):
        # ########## Aeq and Beq Formation ###########
        n_rows = self.n_x
        n_cols = self.n_x
        # Aeq has the same number of rows as equations with a column for each x
        a_eq = zeros((n_rows, n_cols))
        b_eq = zeros(n_rows)
        for j in range(1, self.nb):
            for ph in ["abc", "bca", "cab"]:
                a, b, c = ph[0], ph[1], ph[2]
                if not self.phase_exists(a, j):
                    continue
                a_eq, b_eq = self.add_power_flow_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_voltage_drop_model(a_eq, b_eq, j, a, b, c)
                a_eq, b_eq = self.add_swing_voltage_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_regulator_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_load_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_generator_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_capacitor_model(a_eq, b_eq, j, a)
        return csr_array(a_eq), b_eq

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

    def add_voltage_drop_model(self, a_eq, b_eq, j, a, b, c):
        if self.reg is not None:
            if j in self.reg.tb:
                return a_eq, b_eq
        r, x = self.r, self.x
        aa = "".join(sorted(a + a))
        # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
        ab = "".join(sorted(a + b))
        ac = "".join(sorted(a + c))
        i = self.idx("bi", j, a)[0]  # get the upstream node, i, on branch from i to j
        pij = self.idx("pij", j, a)
        qij = self.idx("qij", j, a)
        pijb = self.idx("pij", j, b)
        qijb = self.idx("qij", j, b)
        pijc = self.idx("pij", j, c)
        qijc = self.idx("qij", j, c)
        vi = self.idx("v", i, a)
        vj = self.idx("v", j, a)
        # Set V equation variable coefficients in a_eq and constants in b_eq
        a_eq[vj, vj] = 1
        a_eq[vj, vi] = -1
        a_eq[vj, pij] = 2 * r[aa][i, j]
        a_eq[vj, qij] = 2 * x[aa][i, j]
        if self.phase_exists(b, j):
            a_eq[vj, pijb] = -r[ab][i, j] + sqrt(3) * x[ab][i, j]
            a_eq[vj, qijb] = -x[ab][i, j] - sqrt(3) * r[ab][i, j]
        if self.phase_exists(c, j):
            a_eq[vj, pijc] = -r[ac][i, j] - sqrt(3) * x[ac][i, j]
            a_eq[vj, qijc] = -x[ac][i, j] + sqrt(3) * r[ac][i, j]
        return a_eq, b_eq

    def add_regulator_model(self, a_eq, b_eq, j, a):
        i = self.idx("bi", j, a)[0]  # get the upstream node, i, on branch from i to j
        vi = self.idx("v", i, a)
        vj = self.idx("v", j, a)

        if self.reg is not None:
            if j in self.reg.tb:
                reg_ratio = get(self.reg[f"ratio_{a}"], j, 1)
                a_eq[vj, vj] = 1
                a_eq[vj, vi] = -1 * reg_ratio**2
                return a_eq, b_eq
        return a_eq, b_eq

    def add_swing_voltage_model(self, a_eq, b_eq, j, a):
        i = self.idx("bi", j, a)[0]  # get the upstream node, i, on branch from i to j
        vi = self.idx("v", i, a)
        # Set V equation variable coefficients in a_eq and constants in b_eq
        if self.bus.bus_type[i] == opf.SWING_BUS:  # Swing bus
            a_eq[vi, vi] = 1
            b_eq[vi] = self.bus.at[i, f"v_{a}"] ** 2
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

    def parse_results(self, x, variable_name: str):
        values = pd.DataFrame(columns=["name", "a", "b", "c"])
        for ph in "abc":
            for j in self.all_buses[ph]:
                values.at[j + 1, "name"] = self.bus.at[j, "name"]
                values.at[j + 1, ph] = x[self.idx(variable_name, j, ph)]
        return values.sort_index()

    def get_device_variables(self, x, variable_map):
        index = np.unique(
            np.r_[
                variable_map["a"].index,
                variable_map["b"].index,
                variable_map["c"].index,
            ]
        )
        bus_id = index + 1
        df = pd.DataFrame(columns=["id", "name", "a", "b", "c"], index=bus_id)
        df.id = bus_id
        df.loc[bus_id, "name"] = self.bus.loc[index, "name"].to_numpy()
        for a in "abc":
            df.loc[variable_map[a].index + 1, a] = x[variable_map[a]]
        return df

    def get_voltages(self, x):
        v_df = self.get_device_variables(x, self.v_map)
        v_df.loc[:, ["a", "b", "c"]] = v_df.loc[:, ["a", "b", "c"]] ** 0.5
        return v_df

    def get_q_gens(self, x):
        return self.get_device_variables(x, self.qg_map)

    # def get_p_gens(self, x):
    #     return self.get_device_variables(x, self.pg_map)

    def get_q_caps(self, x):
        return self.get_device_variables(x, self.qc_map)

    def get_apparent_power_flows(self, x):
        s_df = pd.DataFrame(
            columns=["fb", "tb", "from_name", "to_name", "a", "b", "c"],
            index=range(2, self.nb + 1),
        )
        s_df["a"] = s_df["a"].astype(complex)
        s_df["b"] = s_df["b"].astype(complex)
        s_df["c"] = s_df["c"].astype(complex)
        for ph in "abc":
            fb_idxs = self.x_maps[ph].bi.values
            fb_names = self.bus.name[fb_idxs].to_numpy()
            tb_idxs = self.x_maps[ph].bj.values
            tb_names = self.bus.name[tb_idxs].to_numpy()
            s_df.loc[self.x_maps[ph].bj.values + 1, "fb"] = fb_idxs + 1
            s_df.loc[self.x_maps[ph].bj.values + 1, "tb"] = tb_idxs + 1
            s_df.loc[self.x_maps[ph].bj.values + 1, "from_name"] = fb_names
            s_df.loc[self.x_maps[ph].bj.values + 1, "to_name"] = tb_names
            s_df.loc[self.x_maps[ph].bj.values + 1, ph] = (
                x[self.x_maps[ph].pij] + 1j * x[self.x_maps[ph].qij]
            )
        return s_df

    @property
    def branch_data(self):
        return self.branch

    @property
    def bus_data(self):
        return self.bus

    @property
    def gen_data(self):
        return self.gen

    @property
    def cap_data(self):
        return self.cap

    @property
    def reg_data(self):
        return self.reg
