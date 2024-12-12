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


def _handle_loadshape_input(loadshape_data: pd.DataFrame) -> pd.DataFrame:
    if loadshape_data is None:
        return pd.DataFrame(
            columns=[
                "time",
                "M",
            ]
        )
    loadshape = loadshape_data.sort_values(by="time", ignore_index=True)
    loadshape.index = loadshape.time.to_numpy()
    return loadshape


def _handle_pv_loadshape_input(pv_loadshape_data: pd.DataFrame) -> pd.DataFrame:
    if pv_loadshape_data is None:
        return pd.DataFrame(
            columns=[
                "time",
                "PV",
            ]
        )
    pv_loadshape = pv_loadshape_data.sort_values(by="time", ignore_index=True)
    pv_loadshape.index = pv_loadshape.time.to_numpy()
    return pv_loadshape


def _handle_bat_input(bat_data: pd.DataFrame) -> pd.DataFrame:
    if bat_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "nc_a",
                "nc_b",
                "nc_c",
                "nd_a",
                "nd_b",
                "nd_c",
                "hmax_a",
                "hmax_b",
                "hmax_c",
                "Pb_max_a",
                "Pb_max_b",
                "Pb_max_c",
                "bmin_a",
                "bmin_b",
                "bmin_c",
                "bmax_a",
                "bmax_b",
                "bmax_c",
                "phases",
            ]
        )
    bat = bat_data.sort_values(by="id", ignore_index=True)
    bat.index = bat.id.to_numpy() - 1
    return bat


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


class LinDistModelModular:
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
    bat_data : pd DataFrame
        DataFrame containing battery data
    loadshape_data : pd.DataFrame
        DataFrame containing loadshape multipliers for P values
    pv_loadshape_data : pd.DataFrame
        DataFrame containing PV profile of 1h interval for 24h
    n_steps : int,
        Number of time intervals for multi period optimization. Default is 24.

    """

    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
        loadshape_data: pd.DataFrame = None,
        pv_loadshape_data: pd.DataFrame = None,
        bat_data: pd.DataFrame = None,
        start_step: int = 0,
        n_steps: int = 24,
        delta_t: float = 1,  # hours per step
    ):
        # ~~~~~~~~~~~~~~~~~~~~ Load Data Frames ~~~~~~~~~~~~~~~~~~~~
        self.branch = _handle_branch_input(branch_data)
        self.bus = _handle_bus_input(bus_data)
        self.gen = _handle_gen_input(gen_data)
        self.cap = _handle_cap_input(cap_data)
        self.reg = _handle_reg_input(reg_data)
        self.loadshape = _handle_loadshape_input(loadshape_data)
        self.pv_loadshape = _handle_pv_loadshape_input(pv_loadshape_data)
        self.bat = _handle_bat_input(bat_data)
        self.start_step = start_step
        self.n_steps = n_steps
        self.delta_t = delta_t
        self.SWING = self.bus.loc[self.bus.bus_type == "SWING", "id"].to_numpy()[0] - 1

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
        self.bat_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.bat.shape[0] > 0:
            self.bat_buses = {
                "a": self.bat.loc[self.bat.phases.str.contains("a")].index.to_numpy(),
                "b": self.bat.loc[self.bat.phases.str.contains("b")].index.to_numpy(),
                "c": self.bat.loc[self.bat.phases.str.contains("c")].index.to_numpy(),
            }
        self.n_bats = (
            len(self.bat_buses["a"])
            + len(self.bat_buses["b"])
            + len(self.bat_buses["c"])
        )
        self.controlled_load_buses = {
            "a": self.bus.loc[
                self.bus.bus_type.str.contains(opf.PQ_FREE)
            ].index.to_numpy(),
            "b": self.bus.loc[
                self.bus.bus_type.str.contains(opf.PQ_FREE)
            ].index.to_numpy(),
            "c": self.bus.loc[
                self.bus.bus_type.str.contains(opf.PQ_FREE)
            ].index.to_numpy(),
        }
        # ~~ initialize index pointers ~~
        self.x_maps = {}
        self.v_map = {}
        self.pl_map = {}
        self.ql_map = {}
        self.pg_map = {}
        self.qg_map = {}
        self.qc_map = {}
        self.charge_map = {}
        self.discharge_map = {}
        self.soc_map = {}
        self.n_x = 0
        for t in range(self.start_step, self.start_step + self.n_steps):
            self.x_maps[t], self.n_x = self._variable_tables(self.branch, n_x=self.n_x)
            self.v_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.all_buses
            )
            self.pl_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.all_buses
            )
            self.ql_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.all_buses
            )
            self.pg_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.gen_buses
            )
            self.qg_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.gen_buses
            )
            self.qc_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.cap_buses
            )
            self.charge_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.bat_buses
            )
            self.discharge_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.bat_buses
            )
            self.soc_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.bat_buses
            )
        self.x_maps: dict[int, dict[str, pd.Series]]
        self.v_map: dict[int, dict[str, pd.Series]]
        self.pl_map: dict[int, dict[str, pd.Series]]
        self.ql_map: dict[int, dict[str, pd.Series]]
        self.pg_map: dict[int, dict[str, pd.Series]]
        self.qg_map: dict[int, dict[str, pd.Series]]
        self.qc_map: dict[int, dict[str, pd.Series]]
        self.charge_map: dict[int, dict[str, pd.Series]]
        self.discharge_map: dict[int, dict[str, pd.Series]]
        self.soc_map: dict[int, dict[str, pd.Series]]
        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq ~~~~~~~~~~~~~~~~~~~~
        self._a_eq, self._b_eq = None, None
        self._a_ub, self._b_ub = None, None
        self._bounds = None
        self._bounds_tuple = None

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
        for t in range(self.start_step, self.start_step + self.n_steps):
            x_lim_lower, x_lim_upper = self.add_voltage_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.add_generator_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.add_battery_discharging_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.add_battery_charging_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.add_battery_soc_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.user_added_limits(
                x_lim_lower, x_lim_upper, t=t
            )
        bounds = np.c_[x_lim_lower, x_lim_upper]
        return bounds

    def user_added_limits(self, x_lim_lower, x_lim_upper, t=0):
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
        if t < self.start_step:
            t = self.start_step
        return x_lim_lower, x_lim_upper

    def add_voltage_limits(self, x_lim_lower, x_lim_upper, t=0):
        if t < self.start_step:
            t = self.start_step
        for a in "abc":
            if not self.phase_exists(a):
                continue
            # ~~ v limits ~~:
            x_lim_upper[self.v_map[t][a]] = (
                self.bus.loc[self.v_map[t][a].index, "v_max"] ** 2
            )
            x_lim_lower[self.v_map[t][a]] = (
                self.bus.loc[self.v_map[t][a].index, "v_min"] ** 2
            )
        return x_lim_lower, x_lim_upper

    def add_generator_limits(self, x_lim_lower, x_lim_upper, t=0):
        if t < self.start_step:
            t = self.start_step
        gen_mult = self.pv_loadshape.PV[t]
        for a in "abc":
            if not self.phase_exists(a):
                continue
            q_max_manual = self.gen[f"q{a}_max"]
            q_min_manual = self.gen[f"q{a}_min"]
            s_rated = self.gen[f"s{a}_max"]
            p_out = self.gen[f"p{a}"]
            q_max = ((s_rated**2) - ((p_out*gen_mult)**2)) ** (1 / 2)
            q_min = -1 * q_max
            for j in self.gen_buses[a]:
                mode = self.gen.loc[j, f"{a}_mode"]
                pg = self.idx("pg", j, a, t)
                qg = self.idx("qg", j, a, t)
                # active power bounds
                x_lim_lower[pg] = 0
                x_lim_upper[pg] = p_out[j] * gen_mult
                # reactive power bounds
                if mode == opf.CONSTANT_P:
                    x_lim_lower[qg] = max(q_min[j], q_min_manual[j])
                    x_lim_upper[qg] = min(q_max[j], q_max_manual[j])
                if mode != opf.CONSTANT_P:
                    # reactive power bounds
                    x_lim_lower[qg] = max(-s_rated[j], q_min_manual[j])
                    x_lim_upper[qg] = min(s_rated[j], q_max_manual[j])
        return x_lim_lower, x_lim_upper

    def add_battery_discharging_limits(self, x_lim_lower, x_lim_upper, t=0):
        if t < self.start_step:
            t = self.start_step
        for a in "abc":
            if not self.phase_exists(a):
                continue
            x_lim_lower[self.discharge_map[t][a]] = 0
            x_lim_upper[self.discharge_map[t][a]] = self.bat.loc[self.discharge_map[t][a].index, f"Pb_max_{a}"]
        return x_lim_lower, x_lim_upper

    def add_battery_charging_limits(self, x_lim_lower, x_lim_upper, t=0):
        if t < self.start_step:
            t = self.start_step
        for a in "abc":
            if not self.phase_exists(a):
                continue
            x_lim_lower[self.charge_map[t][a]] = 0
            x_lim_upper[self.charge_map[t][a]] = self.bat.loc[self.charge_map[t][a].index, f"Pb_max_{a}"]
        return x_lim_lower, x_lim_upper

    def add_battery_soc_limits(self, x_lim_lower, x_lim_upper, t=0):
        if t < self.start_step:
            t = self.start_step
        for a in "abc":
            if not self.phase_exists(a):
                continue
            x_lim_upper[self.soc_map[t][a]] = self.bat.loc[self.soc_map[t][a].index, f"bmax_{a}"]
            x_lim_lower[self.soc_map[t][a]] = self.bat.loc[self.soc_map[t][a].index, f"bmin_{a}"]
        return x_lim_lower, x_lim_upper

    @cache
    def branch_into_j(self, var, j, phase, t=0):
        if t < self.start_step:
            t = self.start_step
        idx = self.x_maps[t][phase].loc[self.x_maps[t][phase].bj == j, var].to_numpy()
        return idx[~np.isnan(idx)].astype(int)

    @cache
    def branches_out_of_j(self, var, j, phase, t=0):
        if t < self.start_step:
            t = self.start_step
        idx = self.x_maps[t][phase].loc[self.x_maps[t][phase].bi == j, var].to_numpy()
        return idx[~np.isnan(idx)].astype(int)

    @cache
    def idx(self, var, node_j, phase, t=0):
        if t < self.start_step:
            t = self.start_step
        if var in self.x_maps[t][phase].columns:
            return self.branch_into_j(var, node_j, phase, t=t)
        if var in ["pjk"]:  # indexes of all branch active power out of node j
            return self.branches_out_of_j("pij", node_j, phase, t=t)
        if var in ["qjk"]:  # indexes of all branch reactive power out of node j
            return self.branches_out_of_j("qij", node_j, phase, t=t)
        if var in ["v"]:  # active power generation at node
            return self.v_map[t][phase].get(node_j, [])
        if var in ["pg", "p_gen"]:  # active power generation at node
            return self.pg_map[t][phase].get(node_j, [])
        if var in ["qg", "q_gen"]:  # reactive power generation at node
            return self.qg_map[t][phase].get(node_j, [])
        if var in ["pl", "p_load"]:  # active power load at node
            return self.pl_map[t][phase].get(node_j, [])
        if var in ["ql", "q_load"]:  # reactive power load at node
            return self.ql_map[t][phase].get(node_j, [])
        if var in ["qc", "q_cap"]:  # reactive power injection by capacitor
            return self.qc_map[t][phase].get(node_j, [])
        if var in ["ch", "charge"]:
            return self.charge_map[t][phase].get(node_j, [])
        if var in ["dis", "discharge"]:
            return self.discharge_map[t][phase].get(node_j, [])
        if var in ["soc"]:
            return self.soc_map[t][phase].get(node_j, [])
        ix = self.user_added_idx(var, node_j, phase, t=t)
        if ix is not None:
            return ix
        raise ValueError(f"Variable name, '{var}', not found.")

    def user_added_idx(self, var, node_j, phase, t=0):
        """
        User added index function. Override this function to add custom variables. Return None if `var` is not found.
        Parameters
        ----------
        var : name of variable
        node_j : node index (0 based; bus.id - 1)
        phase : "a", "b", or "c"
        t : integer time step >=0 (default: 0)

        Returns
        -------
        ix : index or list of indices of variable within x-vector or None if `var` is not found.
        """
        if t < self.start_step:
            t = self.start_step
        return None

    @cache
    def phase_exists(self, phase, index: int = None, t=0):
        if t < self.start_step:
            t = self.start_step
        if index is None:
            return self.x_maps[t][phase].shape[0] > 0
        return len(self.idx("bj", index, phase, t=t)) > 0

    def create_model(self):
        # ########## Aeq and Beq Formation ###########
        n_rows = self.n_x
        n_cols = self.n_x
        # Aeq has the same number of rows as equations with a column for each x
        a_eq = zeros((n_rows, n_cols))
        b_eq = zeros(n_rows)
        for t in range(self.start_step, self.start_step + self.n_steps):
            for j in range(1, self.nb):
                for ph in ["abc", "bca", "cab"]:
                    a, b, c = ph[0], ph[1], ph[2]
                    if not self.phase_exists(a, j):
                        continue
                    a_eq, b_eq = self.add_power_flow_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_voltage_drop_model(
                        a_eq, b_eq, j, a, b, c, t=t
                    )
                    a_eq, b_eq = self.add_load_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_generator_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_capacitor_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_battery_model(a_eq, b_eq, j, a, t=t)
        return csr_array(a_eq), b_eq

    def add_power_flow_model(self, a_eq, b_eq, j, phase, t=0):
        if t < self.start_step:
            t = self.start_step
        pij = self.idx("pij", j, phase, t=t)
        qij = self.idx("qij", j, phase, t=t)
        pjk = self.idx("pjk", j, phase, t=t)
        qjk = self.idx("qjk", j, phase, t=t)
        p_discharge_j = self.idx("discharge", j, phase, t=t)
        p_charge_j = self.idx("charge", j, phase, t=t)
        pl = self.idx("pl", j, phase, t=t)
        ql = self.idx("ql", j, phase, t=t)
        pg = self.idx("pg", j, phase, t=t)
        qg = self.idx("qg", j, phase, t=t)
        qc = self.idx("q_cap", j, phase, t=t)
        # Set P equation variable coefficients in a_eq
        a_eq[pij, pij] = 1
        a_eq[pij, pjk] = -1
        a_eq[pij, pl] = -1
        a_eq[pij, pg] = 1
        a_eq[pij, p_discharge_j] = 1
        a_eq[pij, p_charge_j] = -1
        # Set Q equation variable coefficients in a_eq
        a_eq[qij, qij] = 1
        a_eq[qij, qjk] = -1
        a_eq[qij, ql] = -1
        a_eq[qij, qg] = 1
        a_eq[qij, qc] = 1
        return a_eq, b_eq

    def add_voltage_drop_model(self, a_eq, b_eq, j, a, b, c, t=0):
        if t < self.start_step:
            t = self.start_step
        r, x = self.r, self.x
        aa = "".join(sorted(a + a))
        # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
        ab = "".join(sorted(a + b))
        ac = "".join(sorted(a + c))
        i = self.idx("bi", j, a, t=t)[
            0
        ]  # get the upstream node, i, on branch from i to j
        reg_ratio = 1
        if self.reg is not None:
            reg_ratio = get(self.reg[f"ratio_{a}"], j, 1)
        pij = self.idx("pij", j, a, t=t)
        qij = self.idx("qij", j, a, t=t)
        pijb = self.idx("pij", j, b, t=t)
        qijb = self.idx("qij", j, b, t=t)
        pijc = self.idx("pij", j, c, t=t)
        qijc = self.idx("qij", j, c, t=t)
        vi = self.idx("v", i, a, t=t)
        vj = self.idx("v", j, a, t=t)
        # Set V equation variable coefficients in a_eq and constants in b_eq
        if self.bus.bus_type[i] == opf.SWING_BUS:  # Swing bus
            a_eq[vi, vi] = 1
            b_eq[vi] = self.bus.at[i, f"v_{a}"] ** 2

        if self.reg is not None:
            if j in self.reg.tb:
                reg_ratio = self.reg.at[j, f"ratio_{a}"]
                a_eq[vj, vj] = 1
                a_eq[vj, vi] = -1 * reg_ratio**2
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

    def add_generator_model(self, a_eq, b_eq, j, phase, t=0):
        if t < self.start_step:
            t = self.start_step
        a = phase
        p_gen_nom, q_gen_nom = 0, 0
        pv_mult = self.pv_loadshape.PV[t]
        if self.gen is not None:
            p_gen_nom = get(self.gen[f"p{a}"], j, 0)
            q_gen_nom = get(self.gen[f"q{a}"], j, 0)
        # equation indexes
        pg = self.idx("pg", j, a, t=t)
        qg = self.idx("qg", j, a, t=t)
        # Set Generator equation variable coefficients in a_eq
        if get(self.gen[f"{a}_mode"], j, 0) in [opf.CONSTANT_PQ, opf.CONSTANT_P]:
            a_eq[pg, pg] = 1
            b_eq[pg] = p_gen_nom*pv_mult
        if get(self.gen[f"{a}_mode"], j, 0) in [opf.CONSTANT_PQ, opf.CONSTANT_Q]:
            a_eq[qg, qg] = 1
            b_eq[qg] = q_gen_nom
        return a_eq, b_eq

    def add_load_model(self, a_eq, b_eq, j, phase, t=0):
        if t < self.start_step:
            t = self.start_step
        a = phase
        p_load_nom, q_load_nom = 0, 0
        load_mult = self.loadshape.M[t]
        if self.bus.bus_type[j] == opf.PQ_BUS:
            p_load_nom = self.bus[f"pl_{a}"][j] * load_mult
            q_load_nom = self.bus[f"ql_{a}"][j] * load_mult
        # equation indexes
        pl = self.idx("pl", j, a, t=t)
        ql = self.idx("ql", j, a, t=t)
        vj = self.idx("v", j, a, t=t)
        # boundary p and q
        if self.bus.bus_type[j] != opf.PQ_FREE:
            # Set Load equation variable coefficients in a_eq
            a_eq[pl, pl] = 1
            a_eq[pl, vj] = -(self.bus.cvr_p[j] / 2) * p_load_nom
            b_eq[pl] = (1 - (self.bus.cvr_p[j] / 2)) * p_load_nom
            a_eq[ql, ql] = 1
            a_eq[ql, vj] = -(self.bus.cvr_q[j] / 2) * q_load_nom
            b_eq[ql] = (1 - (self.bus.cvr_q[j] / 2)) * q_load_nom
        return a_eq, b_eq

    def add_capacitor_model(self, a_eq, b_eq, j, phase, t=0):
        if t < self.start_step:
            t = self.start_step
        q_cap_nom = 0
        if self.cap is not None:
            q_cap_nom = get(self.cap[f"q{phase}"], j, 0)
        # equation indexes
        vj = self.idx("v", j, phase, t=t)
        qc = self.idx("q_cap", j, phase, t=t)
        a_eq[qc, qc] = 1
        a_eq[qc, vj] = -q_cap_nom
        return a_eq, b_eq

    def add_battery_model(self, a_eq, b_eq, j, phase, t=0):
        if t < self.start_step:
            t = self.start_step
        soc_j = self.idx("soc", j, phase, t=t)
        discharge_j = self.idx("discharge", j, phase, t=t)
        charge_j = self.idx("charge", j, phase, t=t)
        nc = self.bat[f"nc_{phase}"].get(j, 0)
        nd = self.bat[f"nd_{phase}"].get(j, float("inf"))
        soc0 = self.bat[f"b0_{phase}"].get(j, 0)
        # soc0 = self.bat[f"energy_start_{phase}"].get(j, 0)
        dt = 1  # 1 hour time step assumed, currently soc is in units of p_base*1hour (default: 1MWh)
        a_eq[soc_j, discharge_j] = 1 / nd * dt
        a_eq[soc_j, charge_j] = -nc * dt
        a_eq[soc_j, soc_j] = 1
        if t == 0:
            b_eq[soc_j] = soc0
        else:
            soc_prev = self.idx("soc", j, phase, t=t - 1)
            a_eq[soc_j, soc_prev] = -1
        return a_eq, b_eq

    def create_battery_inequality_constraints(self):
        # ########## Aineq and Bineq Formation ###########
        n_inequalities = 1
        n_rows_ineq = n_inequalities * self.n_bats * self.n_steps
        a_ineq = zeros((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        # ineq1 = 0
        # for t in range(self.start_step, self.start_step + self.n_steps):
        #     for a in "abc":
        #         for j in self.bat.index:
        #             if not self.phase_exists(a, j):
        #                 continue
        #             discharge_j = self.idx("discharge", j, a, t=t)
        #             charge_j = self.idx("charge", j, a, t=t)
        #             a_ineq[ineq1, discharge_j] = 1
        #             a_ineq[ineq1, charge_j] = -1
        #             b_ineq[ineq1] = self.bat[f"hmax_{a}"].get(j, 0)
        #             ineq1 += 1
        return a_ineq, b_ineq

    def create_hexagon_constraints(self):
        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        n_inequalities = 6
        n_rows_ineq = n_inequalities * (
            len(np.where(self.gen.a_mode == "CONTROL_PQ")[0])
            + len(np.where(self.gen.a_mode == "CONTROL_PQ")[0])
            + len(np.where(self.gen.a_mode == "CONTROL_PQ")[0])
        ) * self.n_steps
        a_ineq = zeros((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq1 = 0
        ineq2 = 1
        ineq3 = 2
        ineq4 = 3
        ineq5 = 4
        ineq6 = 5
        for t in range(self.start_step, self.start_step + self.n_steps):
            for j in self.gen.index:
                for a in "abc":
                    if not self.phase_exists(a, j):
                        continue
                    if self.gen.loc[j, f"{a}_mode"] != "CONTROL_PQ":
                        continue
                    pg = self.idx("pg", j, a, t=t)
                    qg = self.idx("qg", j, a, t=t)
                    s_rated = self.gen.at[j, f"s{a}_max"]
                    # equation indexes
                    a_ineq[ineq1, pg] = -sqrt(3)
                    a_ineq[ineq1, qg] = -1
                    b_ineq[ineq1] = sqrt(3) * s_rated
                    a_ineq[ineq2, pg] = sqrt(3)
                    a_ineq[ineq2, qg] = 1
                    b_ineq[ineq2] = sqrt(3) * s_rated
                    a_ineq[ineq3, qg] = -1
                    b_ineq[ineq3] = sqrt(3) / 2 * s_rated
                    a_ineq[ineq4, qg] = 1
                    b_ineq[ineq4] = sqrt(3) / 2 * s_rated
                    a_ineq[ineq5, pg] = sqrt(3)
                    a_ineq[ineq5, qg] = -1
                    b_ineq[ineq5] = sqrt(3) * s_rated
                    a_ineq[ineq6, pg] = -sqrt(3)
                    a_ineq[ineq6, qg] = 1
                    b_ineq[ineq6] = -sqrt(3) * s_rated
                    ineq1 += 6
                    ineq2 += 6
                    ineq3 += 6
                    ineq4 += 6
                    ineq5 += 6
                    ineq6 += 6

        return a_ineq, b_ineq

    def create_octagon_constraints(self):
        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        n_inequalities = 5

        n_rows_ineq = n_inequalities * (
            len(np.where(self.gen.a_mode == "CONTROL_PQ")[0])
            + len(np.where(self.gen.a_mode == "CONTROL_PQ")[0])
            + len(np.where(self.gen.a_mode == "CONTROL_PQ")[0])
        ) * self.n_steps
        a_ineq = zeros((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq1 = 0
        ineq2 = 1
        ineq3 = 2
        ineq4 = 3
        ineq5 = 4
        for t in range(self.start_step, self.start_step + self.n_steps):
            for j in self.gen.index:
                for a in "abc":
                    if not self.phase_exists(a, j):
                        continue
                    if self.gen.loc[j, f"{a}_mode"] != "CONTROL_PQ":
                        continue
                    pg = self.idx("pg", j, a, t=t)
                    qg = self.idx("qg", j, a, t=t)
                    s_rated = self.gen.at[j, f"s{a}_max"]
                    # equation indexes
                    a_ineq[ineq1, pg] = sqrt(2)
                    a_ineq[ineq1, qg] = -2 + sqrt(2)
                    b_ineq[ineq1] = sqrt(2) * s_rated
                    a_ineq[ineq2, pg] = sqrt(2)
                    a_ineq[ineq2, qg] = 2 - sqrt(2)
                    b_ineq[ineq2] = sqrt(2) * s_rated
                    a_ineq[ineq3, pg] = -1 + sqrt(2)
                    a_ineq[ineq3, qg] = 1
                    b_ineq[ineq3] = s_rated
                    a_ineq[ineq4, pg] = -1 + sqrt(2)
                    a_ineq[ineq4, qg] = -1
                    b_ineq[ineq4] = s_rated
                    a_ineq[ineq5, pg] = -1
                    b_ineq[ineq5] = 0
                    ineq1 += n_inequalities
                    ineq2 += n_inequalities
                    ineq3 += n_inequalities
                    ineq4 += n_inequalities
                    ineq5 += n_inequalities
        return a_ineq, b_ineq

    def create_inequality_constraints(self):
        a_bat, b_bat = self.create_battery_inequality_constraints()
        a_inv, b_inv = self.create_octagon_constraints()
        a_ub = np.r_[a_bat, a_inv]
        b_ub = np.r_[b_bat, b_inv]
        return csr_array(a_ub), b_ub

    def get_decision_variables(self, x):
        pass

    def get_device_variables(self, x, variable_map):
        df_list = []
        if len(variable_map.keys()) == 0:
            return pd.DataFrame(columns=["id", "name", "t", "a", "b", "c"])
        for t in range(self.start_step, self.start_step + self.n_steps):
            index = np.unique(
                np.r_[
                    variable_map[t]["a"].index,
                    variable_map[t]["b"].index,
                    variable_map[t]["c"].index,
                ]
            )
            bus_id = index + 1
            df = pd.DataFrame(columns=["id", "name", "t", "a", "b", "c"], index=bus_id)
            df.id = bus_id
            df.t = t
            df.loc[bus_id, "name"] = self.bus.loc[index, "name"].to_numpy()
            for a in "abc":
                df.loc[variable_map[t][a].index + 1, a] = x[variable_map[t][a]]
            df_list.append(df)
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        return df

    def get_voltages(self, x):
        v_df = self.get_device_variables(x, self.v_map)
        v_df.loc[:, ["a", "b", "c"]] = v_df.loc[:, ["a", "b", "c"]] ** 0.5
        return v_df

    def get_p_loads(self, x):
        return self.get_device_variables(x, self.pl_map)

    def get_q_loads(self, x):
        return self.get_device_variables(x, self.ql_map)

    def get_q_gens(self, x):
        return self.get_device_variables(x, self.qg_map)

    def get_p_gens(self, x):
        return self.get_device_variables(x, self.pg_map)

    def get_q_caps(self, x):
        return self.get_device_variables(x, self.qc_map)

    def get_p_charge(self, x):
        return self.get_device_variables(x, self.charge_map)

    def get_p_discharge(self, x):
        return self.get_device_variables(x, self.discharge_map)

    def get_soc(self, x):
        return self.get_device_variables(x, self.soc_map)

    def get_apparent_power_flows(self, x):
        df_list = []
        for t in range(self.start_step, self.start_step + self.n_steps):
            s_df = pd.DataFrame(
                columns=["fb", "tb", "from_name", "to_name", "t", "a", "b", "c"],
                index=range(2, self.nb + 1),
            )
            s_df["a"] = s_df["a"].astype(complex)
            s_df["b"] = s_df["b"].astype(complex)
            s_df["c"] = s_df["c"].astype(complex)
            s_df.t = t
            for ph in "abc":
                fb_idxs = self.x_maps[t][ph].bi.values
                fb_names = self.bus.name[fb_idxs].to_numpy()
                tb_idxs = self.x_maps[t][ph].bj.values
                tb_names = self.bus.name[tb_idxs].to_numpy()
                s_df.loc[self.x_maps[t][ph].bj.values + 1, "fb"] = fb_idxs + 1
                s_df.loc[self.x_maps[t][ph].bj.values + 1, "tb"] = tb_idxs + 1
                s_df.loc[self.x_maps[t][ph].bj.values + 1, "from_name"] = fb_names
                s_df.loc[self.x_maps[t][ph].bj.values + 1, "to_name"] = tb_names
                s_df.loc[self.x_maps[t][ph].bj.values + 1, ph] = (
                    x[self.x_maps[t][ph].pij] + 1j * x[self.x_maps[t][ph].qij]
                )
            df_list.append(s_df)
        s_df = pd.concat(df_list, axis=0).reset_index(drop=True)
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

    @property
    def loadshape_data(self):
        return self.loadshape

    @property
    def pv_loadshape_data(self):
        return self.pv_loadshape

    @property
    def bat_data(self):
        return self.bat

    @property
    def a_eq(self):
        if self._a_eq is None:
            self._a_eq, self._b_eq = self.create_model()
        return self._a_eq

    @property
    def b_eq(self):
        if self._b_eq is None:
            self._a_eq, self._b_eq = self.create_model()
        return self._b_eq

    @property
    def a_ub(self):
        if self._a_ub is None:
            self._a_ub, self._b_ub = self.create_inequality_constraints()
        return self._a_ub

    @property
    def b_ub(self):
        if self._b_ub is None:
            self._a_ub, self._b_ub = self.create_inequality_constraints()
        return self._b_ub

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = self.init_bounds()
        if self._bounds_tuple is None:
            self._bounds_tuple = list(map(tuple, self._bounds))
        return self._bounds_tuple

    @property
    def x_min(self):
        if self._bounds is None:
            self._bounds = self.init_bounds()
        return self._bounds[:, 0]

    @property
    def x_max(self):
        if self._bounds is None:
            self._bounds = self.init_bounds()
        return self._bounds[:, 1]

    @property
    def a_ineq(self):
        return self.a_ub

    @property
    def b_ineq(self):
        return self.b_ub