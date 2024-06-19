from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_matrix


# bus_type options
SWING_FREE = "IN"
PQ_FREE = "OUT"
SWING_BUS = "SWING"
PQ_BUS = "PQ"
# generator mode options
CONSTANT_PQ = "CONSTANT_PQ"
CONSTANT_P = "CONSTANT_P"
CONSTANT_Q = "CONSTANT_Q"

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


class LinDistModel:
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
        self.gen_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.gen.shape[0] > 0:
            self.gen_buses = {
                "a": self.gen.loc[self.gen.phases.str.contains("a")].index.to_numpy(),
                "b": self.gen.loc[self.gen.phases.str.contains("b")].index.to_numpy(),
                "c": self.gen.loc[self.gen.phases.str.contains("c")].index.to_numpy(),
            }
        self.cap_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.cap.shape[0] > 0:
            self.cap_buses = {
                "a": self.cap.loc[self.cap.phases.str.contains("a")].index.to_numpy(),
                "b": self.cap.loc[self.cap.phases.str.contains("b")].index.to_numpy(),
                "c": self.cap.loc[self.cap.phases.str.contains("c")].index.to_numpy(),
            }
            nc_a = len(self.cap_buses["a"])
            nc_b = len(self.cap_buses["b"])
            nc_c = len(self.cap_buses["c"])
            self.n_u = nc_a + nc_b + nc_c
        self.load_buses = {
            "a": self.bus.loc[self.bus.phases.str.contains("a")].index.to_numpy(),
            "b": self.bus.loc[self.bus.phases.str.contains("b")].index.to_numpy(),
            "c": self.bus.loc[self.bus.phases.str.contains("c")].index.to_numpy(),
        }
        # ~~ initialize index pointers ~~
        self.x_maps, self.n_x = self._variable_tables(self.branch)
        self.pg_start_phase_idxs, self.n_x = self._add_device_variables(self.n_x, self.gen_buses)
        self.qg_start_phase_idxs, self.n_x = self._add_device_variables(self.n_x, self.gen_buses)
        self.pl_start_phase_idxs, self.n_x = self._add_device_variables(self.n_x, self.load_buses)
        self.ql_start_phase_idxs, self.n_x = self._add_device_variables(self.n_x, self.load_buses)
        self.qc_start_phase_idxs, self.n_x = self._add_device_variables(self.n_x, self.cap_buses)
        self.zc_start_phase_idxs, self.n_x = self._add_device_variables(self.n_x, self.cap_buses)
        self.uc_start_phase_idxs, self.n_x = self._add_device_variables(self.n_x, self.cap_buses)

        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq ~~~~~~~~~~~~~~~~~~~~
        self.a_eq, self.b_eq = self.create_model()
        self.bounds = self.init_bounds(self.bus, self.gen)

    @staticmethod
    def _init_rx(branch):
        row = np.array(np.r_[branch.fb, branch.tb], dtype=int) - 1
        col = np.array(np.r_[branch.tb, branch.fb], dtype=int) - 1
        r = {
            "aa": csr_matrix((np.r_[branch.raa, branch.raa], (row, col))),
            "ab": csr_matrix((np.r_[branch.rab, branch.rab], (row, col))),
            "ac": csr_matrix((np.r_[branch.rac, branch.rac], (row, col))),
            "bb": csr_matrix((np.r_[branch.rbb, branch.rbb], (row, col))),
            "bc": csr_matrix((np.r_[branch.rbc, branch.rbc], (row, col))),
            "cc": csr_matrix((np.r_[branch.rcc, branch.rcc], (row, col))),
        }
        x = {
            "aa": csr_matrix((np.r_[branch.xaa, branch.xaa], (row, col))),
            "ab": csr_matrix((np.r_[branch.xab, branch.xab], (row, col))),
            "ac": csr_matrix((np.r_[branch.xac, branch.xac], (row, col))),
            "bb": csr_matrix((np.r_[branch.xbb, branch.xbb], (row, col))),
            "bc": csr_matrix((np.r_[branch.xbc, branch.xbc], (row, col))),
            "cc": csr_matrix((np.r_[branch.xcc, branch.xcc], (row, col))),
        }
        return r, x

    @staticmethod
    def _variable_tables(branch):
        a_indices = branch.phases.str.contains("a")
        b_indices = branch.phases.str.contains("b")
        c_indices = branch.phases.str.contains("c")
        line_a = branch.loc[a_indices, ["fb", "tb"]].values
        line_b = branch.loc[b_indices, ["fb", "tb"]].values
        line_c = branch.loc[c_indices, ["fb", "tb"]].values
        nl_a = len(line_a)
        nl_b = len(line_b)
        nl_c = len(line_c)
        g = nx.Graph()
        g_a = nx.Graph()
        g_b = nx.Graph()
        g_c = nx.Graph()
        g.add_edges_from(branch[["fb", "tb"]].values.astype(int) - 1)
        g_a.add_edges_from(line_a.astype(int) - 1)
        g_b.add_edges_from(line_b.astype(int) - 1)
        g_c.add_edges_from(line_c.astype(int) - 1)
        t_a = np.array([])
        t_b = np.array([])
        t_c = np.array([])
        if len(g_a.nodes) > 0:
            t_a = np.array(list(nx.dfs_edges(g_a, source=0)))
        if len(g_b.nodes) > 0:
            t_b = np.array(list(nx.dfs_edges(g_b, source=0)))
        if len(g_c.nodes) > 0:
            t_c = np.array(list(nx.dfs_edges(g_c, source=0)))

        p_a_end = 1 * nl_a
        q_a_end = 2 * nl_a
        v_a_end = 3 * nl_a + 1
        p_b_end = v_a_end + 1 * nl_b
        q_b_end = v_a_end + 2 * nl_b
        v_b_end = v_a_end + 3 * nl_b + 1
        p_c_end = v_b_end + 1 * nl_c
        q_c_end = v_b_end + 2 * nl_c
        v_c_end = v_b_end + 3 * nl_c + 1
        df_a = pd.DataFrame(columns=["bi", "bj", "pij", "qij", "vi", "vj"])
        df_b = pd.DataFrame(columns=["bi", "bj", "pij", "qij", "vi", "vj"])
        df_c = pd.DataFrame(columns=["bi", "bj", "pij", "qij", "vi", "vj"])
        if len(g_a.nodes) > 0:
            df_a = pd.DataFrame(
                {
                    "bi": t_a[:, 0],
                    "bj": t_a[:, 1],
                    "pij": np.array([i for i in range(p_a_end)]),
                    "qij": np.array([i for i in range(p_a_end, q_a_end)]),
                    "vi": np.zeros_like(t_a[:, 0]),
                    "vj": np.array([i for i in range(q_a_end + 1, v_a_end)]),
                },
                dtype=np.int32,
            )
            df_a.loc[0, "vi"] = df_a.at[0, "vj"] - 1
            for i in df_a.bi.values[1:]:
                df_a.loc[df_a.loc[:, "bi"] == i, "vi"] = df_a.loc[
                    df_a.bj == i, "vj"
                ].values[0]
        if len(g_b.nodes) > 0:
            df_b = pd.DataFrame(
                {
                    "bi": t_b[:, 0],
                    "bj": t_b[:, 1],
                    "pij": np.array([i for i in range(v_a_end, p_b_end)]),
                    "qij": np.array([i for i in range(p_b_end, q_b_end)]),
                    "vi": np.zeros_like(t_b[:, 0]),
                    "vj": np.array([i for i in range(q_b_end + 1, v_b_end)]),
                },
                dtype=np.int32,
            )
            df_b.loc[0, "vi"] = df_b.at[0, "vj"] - 1
            for i in df_b.bi.values[1:]:
                df_b.loc[df_b.loc[:, "bi"] == i, "vi"] = df_b.loc[
                    df_b.bj == i, "vj"
                ].values[0]
        if len(g_c.nodes) > 0:
            df_c = pd.DataFrame(
                {
                    "bi": t_c[:, 0],
                    "bj": t_c[:, 1],
                    "pij": [i for i in range(v_b_end, p_c_end)],
                    "qij": [i for i in range(p_c_end, q_c_end)],
                    "vi": np.zeros_like(t_c[:, 0]),
                    "vj": [i for i in range(q_c_end + 1, v_c_end)],
                },
                dtype=np.int32,
            )
            df_c.loc[0, "vi"] = df_c.at[0, "vj"] - 1
            for i in df_c.bi.values[1:]:
                df_c.loc[df_c.loc[:, "bi"] == i, "vi"] = df_c.loc[
                    df_c.bj == i, "vj"
                ].values[0]
        n_x = v_c_end  # start with the largest index so far

        x_maps = {"a": df_a, "b": df_b, "c": df_c}
        return x_maps, n_x


    @staticmethod
    def _add_device_variables(n_x: int, device_buses: dict):
        n_a = len(device_buses["a"])
        n_b = len(device_buses["b"])
        n_c = len(device_buses["c"])
        start_phase_idxs = {
            "a": n_x,
            "b": n_x + n_a,
            "c": n_x + n_a + n_b,
        }
        n_x = n_x + n_a + n_b + n_c
        return start_phase_idxs, n_x

    def init_bounds(self, bus, gen):
        default = 100e3  # Default for unbounded variables.
        x_maps = self.x_maps
        # ~~~~~~~~~~ x limits ~~~~~~~~~~
        x_lim_lower = np.ones(self.n_x) * -default
        x_lim_upper = np.ones(self.n_x) * default
        for a in "abc":
            if self.phase_exists(a):
                x_lim_lower[x_maps[a].loc[:, "pij"]] = -default  # P
                x_lim_upper[x_maps[a].loc[:, "pij"]] = default  # P
                x_lim_lower[x_maps[a].loc[:, "qij"]] = -default  # Q
                x_lim_upper[x_maps[a].loc[:, "qij"]] = default  # Q
                # ~~ v limits ~~:
                i_root = list(set(x_maps["a"].bi) - set(x_maps["a"].bj))[0]
                i_v_swing = (
                    x_maps[a]
                    .loc[x_maps[a].loc[:, "bi"] == i_root, "vi"]
                    .to_numpy()[0]
                )
                x_lim_lower[i_v_swing] = bus.loc[i_root, "v_min"] ** 2
                x_lim_upper[i_v_swing] = bus.loc[i_root, "v_max"] ** 2
                x_lim_lower[x_maps[a].loc[:, "vj"]] = (
                    bus.loc[x_maps[a].loc[:, "bj"], "v_min"] ** 2
                )
                x_lim_upper[x_maps[a].loc[:, "vj"]] = (
                    bus.loc[x_maps[a].loc[:, "bj"], "v_max"] ** 2
                )
                for j in self.gen_buses[a]:
                    i_p = self.idx("pg", j, a)
                    i_q = self.idx("qg", j, a)
                    q_max_manual = gen[f"q{a}_max"][j]
                    q_min_manual = gen[f"q{a}_min"][j]
                    s_rated: pd.Series = gen[f"s{a}_max"]
                    p_out: pd.Series = gen[f"p{a}"]
                    # active power bounds
                    x_lim_lower[i_p] = 0
                    x_lim_upper[i_p] = p_out[j]
                    # reactive power bounds
                    q_min: pd.Series = -(((s_rated**2) - (p_out**2)) ** (1 / 2))
                    q_max: pd.Series = ((s_rated**2) - (p_out**2)) ** (1 / 2)
                    x_lim_lower[i_q] = max(q_min[j], q_min_manual)
                    x_lim_upper[i_q] = min(q_max[j], q_max_manual)
        bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
        return bounds

    @cache
    def branch_into_j(self, var, j, phase):
        return self.x_maps[phase].loc[self.x_maps[phase].bj == j, var].to_numpy()

    @cache
    def branches_out_of_j(self, var, j, phase):
        return self.x_maps[phase].loc[self.x_maps[phase].bi == j, var].to_numpy()

    @cache
    def idx(self, var, node_j, phase):
        if var == "q_cap":  # active power generation at node
            if node_j in set(self.cap_buses[phase]):
                return (
                    self.qc_start_phase_idxs[phase]
                    + np.where(self.cap_buses[phase] == node_j)[0]
                )
            return []
        if var == "z_c":
            if node_j in set(self.cap_buses[phase]):
                return (
                    self.zc_start_phase_idxs[phase]
                    + np.where(self.cap_buses[phase] == node_j)[0]
                )
            return []
        if var == "u_c":
            if node_j in set(self.cap_buses[phase]):
                return (
                    self.uc_start_phase_idxs[phase]
                    + np.where(self.cap_buses[phase] == node_j)[0]
                )
            return []
        if var == "pg":  # active power generation at node
            if node_j in set(self.gen_buses[phase]):
                return (
                    self.pg_start_phase_idxs[phase]
                    + np.where(self.gen_buses[phase] == node_j)[0]
                )
            return []
        if var == "qg":  # reactive power generation at node
            if node_j in set(self.gen_buses[phase]):
                return (
                    self.qg_start_phase_idxs[phase]
                    + np.where(self.gen_buses[phase] == node_j)[0]
                )
            return []
        if var == "pl":  # active power load at node
            if node_j in set(self.load_buses[phase]):
                return (
                    self.pl_start_phase_idxs[phase]
                    + np.where(self.load_buses[phase] == node_j)[0]
                )
        if var == "ql":  # reactive power load at node
            if node_j in set(self.load_buses[phase]):
                return (
                    self.ql_start_phase_idxs[phase]
                    + np.where(self.load_buses[phase] == node_j)[0]
                )
        if var in ["pjk"]:  # indexes of all branch active power out of node j
            return self.branches_out_of_j("pij", node_j, phase)
        if var in ["qjk"]:  # indexes of all branch reactive power out of node j
            return self.branches_out_of_j("qij", node_j, phase)
        # self.user_added_idx(var, node_j, phase)
        return self.branch_into_j(var, node_j, phase)

    @cache
    def phase_exists(self, phase, index: int = None):
        if index is None:
            return self.x_maps[phase].shape[0] > 0
        return len(self.idx("bj", index, phase)) > 0

    def create_model(self):
        r, x = self.r, self.x
        bus = self.bus

        # ########## Aeq and Beq Formation ###########
        n_rows = self.n_x
        n_cols = self.n_x
        # Aeq has the same number of rows as equations with a column for each x
        a_eq = zeros((n_rows, n_cols))
        b_eq = zeros(n_rows)
        for j in range(1, self.nb):
            for ph in ["abc", "bca", "cab"]:
                a, b, c = ph[0], ph[1], ph[2]
                aa = "".join(sorted(a + a))
                # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
                ab = "".join(sorted(a + b))
                ac = "".join(sorted(a + c))
                if not self.phase_exists(a, j):
                    continue
                reg_ratio = 1
                if self.reg is not None:
                    reg_ratio = get(self.reg[f"ratio_{a}"], j, 1)
                # equation indexes
                pij = self.idx("pij", j,  a)
                qij = self.idx("qij", j,  a)
                pijb = self.idx("pij", j,  b)
                qijb = self.idx("qij", j,  b)
                pijc = self.idx("pij", j,  c)
                qijc = self.idx("qij", j,  c)
                pjk = self.idx("pjk", j, a)
                qjk = self.idx("qjk", j, a)
                vi = self.idx("vi", j, a)
                vj = self.idx("vj", j, a)
                pl = self.idx("pl", j, a)
                ql = self.idx("ql", j, a)
                pg = self.idx("pg", j, a)
                qg = self.idx("qg", j, a)
                qc = self.idx("q_cap", j, a)
                # Set P equation variable coefficients in a_eq
                a_eq[pij, pij] = 1
                a_eq[pij, pjk] = -1
                a_eq[pij, pl] = -1
                a_eq[pij, pg] = 1
                # Set Q equation variable coefficients in a_eq
                a_eq[qij, qij] = 1
                a_eq[qij, qjk] = -1
                a_eq[qij, ql] = -1
                a_eq[qij, qg] = 1
                a_eq[qij, qc] = 1
                # Set V equation variable coefficients in a_eq and constants in b_eq
                i = self.idx("bi", j, a)[0]  # get the upstream node, i, on branch from i to j
                if bus.bus_type[i] == SWING_BUS:  # Swing bus
                    a_eq[vi, vi] = 1
                    b_eq[vi] = bus.at[i, f"v_{a}"] ** 2
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
                a_eq, b_eq = self.add_load_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_generator_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_capacitor_model(a_eq, b_eq, j, a)
        return a_eq, b_eq


    def add_generator_model(self, a_eq, b_eq, j, phase):
        a = phase
        p_gen_nom, q_gen_nom = 0, 0
        if self.gen is not None:
            p_gen_nom = get(self.gen[f"p{a}"], j, 0)
            q_gen_nom = get(self.gen[f"q{a}"], j, 0)
        # equation indexes
        pg = self.idx("pg", j, a)
        qg = self.idx("qg", j, a)
        # Set Generator equation variable coefficients in a_eq
        if get(self.gen[f"{a}_mode"], j, 0) in [CONSTANT_PQ, CONSTANT_P]:
            a_eq[pg, pg] = 1
            b_eq[pg] = p_gen_nom
        if get(self.gen[f"{a}_mode"], j, 0) in [CONSTANT_PQ, CONSTANT_Q]:
            a_eq[qg, qg] = 1
            b_eq[qg] = q_gen_nom
        return a_eq, b_eq

    def add_load_model(self, a_eq, b_eq, j, phase):
        a = phase
        p_load_nom, q_load_nom = 0, 0
        if self.bus.bus_type[j] == PQ_BUS:
            p_load_nom = self.bus[f"pl_{a}"][j]
            q_load_nom = self.bus[f"ql_{a}"][j]
        # equation indexes
        pl = self.idx("pl", j, a)
        ql = self.idx("ql", j, a)
        vj = self.idx("vj", j, a)
        # boundary p and q
        if self.bus.bus_type[j] != PQ_FREE:
            # Set Load equation variable coefficients in a_eq
            a_eq[pl, pl] = 1
            a_eq[pl, vj] = -(self.bus.cvr_p[j] / 2) * p_load_nom
            b_eq[pl] = (1 - (self.bus.cvr_p[j] / 2)) * p_load_nom
            a_eq[ql, ql] = 1
            a_eq[ql, vj] = -(self.bus.cvr_q[j] / 2) * q_load_nom
            b_eq[ql] = (1 - (self.bus.cvr_q[j] / 2)) * q_load_nom
        return a_eq, b_eq

    def add_capacitor_model(self, a_eq, b_eq, j, phase):
        q_cap_nom = 0
        if self.cap is not None:
            q_cap_nom = get(self.cap[f"q{phase}"], j, 0)
        # equation indexes
        vj = self.idx("vj", j, phase)
        qc = self.idx("q_cap", j, phase)
        a_eq[qc, qc] = 1
        a_eq[qc, vj] = -q_cap_nom
        return a_eq, b_eq


    def parse_results(self, x, variable_name: str):
        values = pd.DataFrame(columns=["name", "a", "b", "c"])
        for ph in "abc":
            for j in self.load_buses[ph]:
                values.at[j + 1, "name"] = self.bus.at[j, "name"]
                values.at[j + 1, ph] = x[self.idx(variable_name, j, ph)]
        return values

    def get_decision_variables(self, x):
        decision_variables = pd.DataFrame(columns=["name", "a", "b", "c"])
        for ph in "abc":
            for j in self.gen_buses[ph]:
                decision_variables.at[j + 1, "name"] = self.bus.at[j, "name"]
                decision_variables.at[j + 1, ph] = x[self.idx("qg", j, ph)]
        return decision_variables


    def get_p_gens(self, x):
        decision_variables = pd.DataFrame(columns=["name", "a", "b", "c"])
        for ph in "abc":
            for j in self.gen_buses[ph]:
                decision_variables.at[j + 1, "name"] = self.bus.at[j, "name"]
                decision_variables.at[j + 1, ph] = x[self.idx("pg", j, ph)]
        return decision_variables

    def get_q_gens(self, x):
        decision_variables = pd.DataFrame(columns=["name", "a", "b", "c"])
        for ph in "abc":
            for j in self.gen_buses[ph]:
                decision_variables.at[j + 1, "name"] = self.bus.at[j, "name"]
                decision_variables.at[j + 1, ph] = x[self.idx("qg", j, ph)]
        return decision_variables

    def get_voltages(self, x):
        v_df = pd.DataFrame(
            columns=["name", "a", "b", "c"],
            index=np.array(range(1, self.nb + 1)),
        )
        v_df["name"] = self.bus["name"].to_numpy()
        for ph in "abc":
            if not self.phase_exists(ph):
                v_df.loc[:, ph] = 0.0
                continue
            v_df.loc[1, ph] = np.sqrt(x[self.x_maps[ph].vi[0]].astype(np.float64))
            v_df.loc[self.x_maps[ph].bj.values + 1, ph] = np.sqrt(
                x[self.x_maps[ph].vj.values].astype(np.float64)
            )
        return v_df

    def get_apparent_power_flows(self, x):
        s_df = pd.DataFrame(
            columns=["fb", "tb", "a", "b", "c"], index=range(2, self.nb + 1)
        )
        s_df["a"] = s_df["a"].astype(complex)
        s_df["b"] = s_df["b"].astype(complex)
        s_df["c"] = s_df["c"].astype(complex)
        for ph in "abc":
            fb_idxs = self.x_maps[ph].bi.values
            fb_names = self.bus.name[fb_idxs].to_numpy()
            tb_idxs = self.x_maps[ph].bj.values
            tb_names = self.bus.name[tb_idxs].to_numpy()
            s_df.loc[self.x_maps[ph].bj.values + 1, "fb"] = fb_names
            s_df.loc[self.x_maps[ph].bj.values + 1, "tb"] = tb_names
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
