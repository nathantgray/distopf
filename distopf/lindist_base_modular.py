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
            ]
        )
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
            # check consistency
            if any(abs(reg[f"ratio_{ph}"]) - (1 + 0.00625 * reg[f"tap_{ph}"]) > 1e-6):
                raise ValueError(
                    f"Regulator taps and ratio are inconsistent on phase {ph}!"
                )
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
        self.der_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        self.cap_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.gen.shape[0] > 0:
            self.der_buses = {
                "a": self.gen.loc[self.gen.phases.str.contains("a")].index.to_numpy(),
                "b": self.gen.loc[self.gen.phases.str.contains("b")].index.to_numpy(),
                "c": self.gen.loc[self.gen.phases.str.contains("c")].index.to_numpy(),
            }
        if self.cap.shape[0] > 0:
            self.cap_buses = {
                "a": self.cap.loc[self.cap.phases.str.contains("a")].index.to_numpy(),
                "b": self.cap.loc[self.cap.phases.str.contains("b")].index.to_numpy(),
                "c": self.cap.loc[self.cap.phases.str.contains("c")].index.to_numpy(),
            }
        self.load_buses = {
            "a": self.bus.loc[self.bus.bus_type.str.contains(PQ_BUS)].index.to_numpy(),
            "b": self.bus.loc[self.bus.bus_type.str.contains(PQ_BUS)].index.to_numpy(),
            "c": self.bus.loc[self.bus.bus_type.str.contains(PQ_BUS)].index.to_numpy(),
        }
        self.controlled_load_buses = {
            "a": self.bus.loc[self.bus.bus_type.str.contains(PQ_FREE)].index.to_numpy(),
            "b": self.bus.loc[self.bus.bus_type.str.contains(PQ_FREE)].index.to_numpy(),
            "c": self.bus.loc[self.bus.bus_type.str.contains(PQ_FREE)].index.to_numpy(),
        }
        # ~~ initialize index pointers ~~
        self.x_maps, self.ctr_var_start_idx = self._variable_tables(self.branch)
        (
            self.pg_start_phase_idxs,
            self.qg_start_phase_idxs,
            self.p_load_start_phase_idxs,
            self.q_load_start_phase_idxs,
            self.cap_start_phase_idxs,
            self.z_c_start_phase_idxs,
            self.n_x,
        ) = self._control_variables(self.ctr_var_start_idx)

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
        ctr_var_start_idx = v_c_end  # start with the largest index so far

        x_maps = {"a": df_a, "b": df_b, "c": df_c}
        return x_maps, ctr_var_start_idx

    def _control_variables(self, ctr_var_start_idx):
        ctr_var_start_idx = int(ctr_var_start_idx)
        ng_a = len(self.der_buses["a"])
        ng_b = len(self.der_buses["b"])
        ng_c = len(self.der_buses["c"])
        nc_a = len(self.cap_buses["a"])
        nc_b = len(self.cap_buses["b"])
        nc_c = len(self.cap_buses["c"])
        pg_start_phase_idxs = {
            "a": ctr_var_start_idx,
            "b": ctr_var_start_idx + ng_a,
            "c": ctr_var_start_idx + ng_a + ng_b,
        }
        qg_start_idx = ctr_var_start_idx + ng_a + ng_b + ng_c
        qg_start_phase_idxs = {
            "a": qg_start_idx,
            "b": qg_start_idx + ng_a,
            "c": qg_start_idx + ng_a + ng_b,
        }
        load_start_idx = qg_start_idx + ng_a + ng_b + ng_c
        n_controlled_load_nodes = sum(self.bus.bus_type == PQ_FREE)
        n_load_nodes = sum(self.bus.bus_type == PQ_BUS)
        p_load_start_phase_idxs = {
            "a": load_start_idx,
            "b": load_start_idx + n_load_nodes,
            "c": load_start_idx + n_load_nodes * 2,
        }
        q_load_start_phase_idxs = {
            "a": load_start_idx + n_load_nodes * 3,
            "b": load_start_idx + n_load_nodes * 4,
            "c": load_start_idx + n_load_nodes * 5,
        }
        cap_start_idx = load_start_idx + n_load_nodes * 6
        cap_start_phase_idxs = {
            "a": cap_start_idx,
            "b": cap_start_idx + nc_a,
            "c": cap_start_idx + nc_a + nc_b,
        }
        z_c_start_idx = cap_start_idx + nc_a + nc_b + nc_c
        z_c_start_phase_idxs = {
            "a": z_c_start_idx,
            "b": z_c_start_idx + nc_a,
            "c": z_c_start_idx + nc_a + nc_b,
        }
        n_x = load_start_idx + z_c_start_idx + nc_a + nc_b + nc_c
        return (pg_start_phase_idxs,
                qg_start_phase_idxs,
                p_load_start_phase_idxs,
                q_load_start_phase_idxs,
                cap_start_phase_idxs,
                z_c_start_phase_idxs,
                n_x
                )

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
                for idx, j in enumerate(self.der_buses[a]):
                    i_p = self.pg_start_phase_idxs[a] + idx
                    i_q = self.qg_start_phase_idxs[a] + idx
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
                    self.cap_start_phase_idxs[phase]
                    + np.where(self.cap_buses[phase] == node_j)[0]
                )
            return []
        if var == "z_c":
            if node_j in set(self.cap_buses[phase]):
                return (
                    self.z_c_start_phase_idxs[phase]
                    + np.where(self.cap_buses[phase] == node_j)[0]
                )
            return []
        if var == "u_c":
            if node_j in set(self.cap_buses[phase]):
                return (
                    self.z_c_start_phase_idxs[phase]
                    + np.where(self.cap_buses[phase] == node_j)[0]
                )
            return []

        if var == "pg":  # active power generation at node
            if node_j in set(self.der_buses[phase]):
                return (
                    self.pg_start_phase_idxs[phase]
                    + np.where(self.der_buses[phase] == node_j)[0]
                )
            return []
        if var == "qg":  # reactive power generation at node
            if node_j in set(self.der_buses[phase]):
                return (
                    self.qg_start_phase_idxs[phase]
                    + np.where(self.der_buses[phase] == node_j)[0]
                )
            return []
        if var == "pl":  # active power exported at node (not root node)
            if node_j in set(self.load_buses[phase]):
                return (
                    self.p_load_start_phase_idxs[phase]
                    + np.where(self.load_buses[phase] == node_j)[0]
                )
        if var == "ql":  # reactive power exported at node (not root node)
            if node_j in set(self.load_buses[phase]):
                return (
                    self.q_load_start_phase_idxs[phase]
                    + np.where(self.load_buses[phase] == node_j)[0]
                )
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
        n_col = self.n_x
        a_eq = zeros(
            (n_rows, n_col)
        )  # Aeq has the same number of rows as equations with a column for each x
        b_eq = zeros(n_rows)
        for j in range(1, self.nb):

            def col(var, phase):
                return self.idx(var, j, phase)

            def row(var, phase):
                return self.idx(var, j, phase)

            def children(var, phase):
                return self.branches_out_of_j(var, j, phase)

            for ph in ["abc", "bca", "cab"]:
                a, b, c = ph[0], ph[1], ph[2]
                aa = "".join(sorted(a + a))
                ab = "".join(
                    sorted(a + b)
                )  # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
                ac = "".join(sorted(a + c))
                if not self.phase_exists(a, j):
                    continue
                p_load_nom, q_load_nom = 0, 0
                reg_ratio = 1
                q_cap_nom = 0
                if bus.bus_type[j] == "PQ":
                    p_load_nom = self.bus[f"pl_{a}"][j]
                    q_load_nom = self.bus[f"ql_{a}"][j]
                p_gen_nom = get(self.gen[f"p{a}"], j, 0)
                q_gen_nom = get(self.gen[f"q{a}"], j, 0)
                if self.cap is not None:
                    q_cap_nom = get(self.cap[f"q{a}"], j, 0)
                if self.reg is not None:
                    reg_ratio = get(self.reg[f"ratio_{a}"], j, 1)
                # equation indexes
                p_eqn = row("pij", a)
                q_eqn = row("qij", a)
                v_eqn = row("vj", a)
                pl_eqn = row("pl", a)
                ql_eqn = row("ql", a)
                pg_eqn = row("pg", a)
                qg_eqn = row("qg", a)
                qc_eqn = row("q_cap", a)
                # Set P equation variable coefficients in a_eq
                a_eq[p_eqn, col("pij", a)] = 1
                a_eq[p_eqn, children("pij", a)] = -1
                a_eq[p_eqn, col("pl", a)] = -1
                a_eq[p_eqn, col("pg", a)] = 1
                # Set Q equation variable coefficients in a_eq
                a_eq[q_eqn, col("qij", a)] = 1
                a_eq[q_eqn, children("qij", a)] = -1
                a_eq[q_eqn, col("ql", a)] = -1
                a_eq[q_eqn, col("qg", a)] = 1
                a_eq[q_eqn, col("q_cap", a)] = 1
                # Set V equation variable coefficients in a_eq and constants in b_eq
                i = self.idx("bi", j, a)[0]
                if bus.bus_type[i] == "SWING":  # Swing bus
                    a_eq[row("vi", a), col("vi", a)] = 1
                    b_eq[row("vi", a)] = (
                        bus.loc[bus.bus_type == "SWING", f"v_{a}"][0] ** 2
                    )
                a_eq[v_eqn, col("vj", a)] = 1
                a_eq[v_eqn, col("vi", a)] = -1 * reg_ratio**2
                a_eq[v_eqn, col("pij", a)] = 2 * r[aa][i, j]
                a_eq[v_eqn, col("qij", a)] = 2 * x[aa][i, j]
                if self.phase_exists(b, j):
                    a_eq[v_eqn, col("pij", b)] = -r[ab][i, j] + sqrt(3) * x[ab][i, j]
                    a_eq[v_eqn, col("qij", b)] = -x[ab][i, j] - sqrt(3) * r[ab][i, j]
                if self.phase_exists(c, j):
                    a_eq[v_eqn, col("pij", c)] = -r[ac][i, j] - sqrt(3) * x[ac][i, j]
                    a_eq[v_eqn, col("qij", c)] = -x[ac][i, j] + sqrt(3) * r[ac][i, j]

                # boundary p and q
                if bus.bus_type[j] != PQ_FREE:
                    # Set Load equation variable coefficients in a_eq
                    a_eq[pl_eqn, col("pl", a)] = 1
                    a_eq[pl_eqn, col("vj", a)] = -(bus.cvr_p[j] / 2) * p_load_nom
                    b_eq[pl_eqn] = (1 - (bus.cvr_p[j] / 2)) * p_load_nom
                    a_eq[ql_eqn, col("ql", a)] = 1
                    a_eq[ql_eqn, col("vj", a)] = -(bus.cvr_q[j] / 2) * q_load_nom
                    b_eq[ql_eqn] = (1 - (bus.cvr_q[j] / 2)) * q_load_nom
                # Set Generator equation variable coefficients in a_eq
                if get(self.gen[f"{a}_mode"], j, 0) in [0, 1]:
                    a_eq[pg_eqn, col("pg", a)] = 1
                    b_eq[pg_eqn] = p_gen_nom
                if get(self.gen[f"{a}_mode"], j, 0) in [0, 2]:
                    a_eq[qg_eqn, col("qg", a)] = 1
                    b_eq[qg_eqn] = q_gen_nom
                # Set Capacitor equation variable coefficients in a_eq
                a_eq[qc_eqn, col("q_cap", a)] = 1
                a_eq[qc_eqn, col("vj", a)] = -q_cap_nom

        return a_eq, b_eq

    def get_decision_variables(self, x):
        ng_a = len(self.der_buses["a"])
        ng_b = len(self.der_buses["b"])
        ng_c = len(self.der_buses["c"])
        ng = dict(a=ng_a, b=ng_b, c=ng_c)
        i_p = self.pg_start_phase_idxs
        i_q = self.pg_start_phase_idxs
        decision_variables = pd.DataFrame(columns=["name", "a", "b", "c"])
        if self.gen_data.shape[0] == 1 and self.gen_data.index[0] == -1:
            return decision_variables
        for ph in "abc":
            for i_gen in range(ng[ph]):
                i = self.der_buses[ph][i_gen]
                decision_variables.at[i + 1, "name"] = self.bus.at[i, "name"]
                decision_variables.at[i + 1, ph] = [i_q[ph] + i_gen]
        return decision_variables

    def get_gens(self, x):
        ng_a = len(self.der_buses["a"])
        ng_b = len(self.der_buses["b"])
        ng_c = len(self.der_buses["c"])
        ng = dict(a=ng_a, b=ng_b, c=ng_c)
        i_p = self.pg_start_phase_idxs
        i_q = self.pg_start_phase_idxs
        decision_variables = pd.DataFrame(columns=["name", "a", "b", "c"])
        if self.gen_data.shape[0] == 1 and self.gen_data.index[0] == -1:
            return decision_variables
        for ph in "abc":
            for i_gen in range(ng[ph]):
                i = self.der_buses[ph][i_gen]
                decision_variables.at[i + 1, "name"] = self.bus.at[i, "name"]
                decision_variables.at[i + 1, ph] = x[i_p[ph] + i_gen] + 1j*[i_q[ph] + i_gen]
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
