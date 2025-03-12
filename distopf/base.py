from functools import cache
from typing import Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_array, lil_array  # type: ignore
import distopf as opf
import warnings
from distopf.utils import (
    handle_branch_input,
    handle_bus_input,
    handle_gen_input,
    handle_cap_input,
    handle_reg_input,
    get,
)


class BaseModel:
    """
    LinDistFlow Model base class.

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
        # ~~~~~~~~~~~~~~~~~~~~ Load Data Frames ~~~~~~~~~~~~~~~~~~~~
        self.branch = handle_branch_input(branch_data)
        self.bus = handle_bus_input(bus_data)
        self.gen = handle_gen_input(gen_data)
        self.cap = handle_cap_input(cap_data)
        self.reg = handle_reg_input(reg_data)
        self.branch_data = self.branch
        self.bus_data = self.bus
        self.gen_data = self.gen
        self.cap_data = self.cap
        self.reg_data = self.reg

        # ~~~~~~~~~~~~~~~~~~~~ prepare data ~~~~~~~~~~~~~~~~~~~~
        self.nb = len(self.bus.id)
        self.r, self.x = self._init_rx(self.branch)
        self.swing_bus = self.bus.loc[self.bus.bus_type == "SWING"].index[0]
        self.all_buses = {
            "a": self.bus.loc[self.bus.phases.str.contains("a")].index.to_numpy(),
            "b": self.bus.loc[self.bus.phases.str.contains("b")].index.to_numpy(),
            "c": self.bus.loc[self.bus.phases.str.contains("c")].index.to_numpy(),
        }
        self.load_buses = {
            "a": self.all_buses["a"][np.where(self.all_buses["a"] != self.swing_bus)],
            "b": self.all_buses["b"][np.where(self.all_buses["b"] != self.swing_bus)],
            "c": self.all_buses["c"][np.where(self.all_buses["c"] != self.swing_bus)],
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
        self.n_x = 0
        self.x_maps: dict[str, pd.DataFrame] = {}
        self.v_map: dict[str, pd.Series] = {}
        self.pg_map: dict[str, pd.Series] = {}
        self.qg_map: dict[str, pd.Series] = {}
        self.qc_map: dict[str, pd.Series] = {}
        self.pl_map: dict[str, pd.Series] = {}
        self.ql_map: dict[str, pd.Series] = {}
        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq ~~~~~~~~~~~~~~~~~~~~
        self.a_eq, self.b_eq = csr_array([0]), zeros(0)
        self.a_ub, self.b_ub = csr_array([0]), zeros(0)
        self.bounds: np.ndarray = zeros(0)
        self.bounds_tuple: list[tuple] = []
        self.x_min: np.ndarray = zeros(0)
        self.x_max: np.ndarray = zeros(0)

    @staticmethod
    def _init_rx(branch):
        """
        Initializes resistance (`r`) and reactance (`x`) data for network branches
        using sparse matrix representations.

        Parameters
        ----------
        branch : pd.DataFrame
            DataFrame containing branch information.

        Returns
        -------
        r, x : dict
            Dictionaries of csr_arrays for resistance and reactance matrices indexed
            by phase pairs (e.g., 'aa', 'ab', ...).
        """
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


class LinDistBase(BaseModel):
    """
    LinDistFlow Model base class for linear power flow modeling.

    This class represents a linearized distribution model used for calculating
    power flows, voltages, and other system properties in a distribution network
    using the linearized branch-flow formulation from [1]. The model is composed of several power system components
    such as buses, branches, generators, capacitors, and regulators.

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

    References
    ----------
    [1] R. R. Jha, A. Dubey, C.-C. Liu, and K. P. Schneider,
    “Bi-Level Volt-VAR Optimization to Coordinate Smart Inverters
    With Voltage Control Devices,”
    IEEE Trans. Power Syst., vol. 34, no. 3, pp. 1801–1813,
    May 2019, doi: 10.1109/TPWRS.2018.2890613.

    Examples
    --------
    This example demonstrates how to set up and solve a linear distribution flow model
    using a provided case, and visualize the results.

    >>> import distopf as opf
    >>> # Prepare the case data
    >>> case = opf.DistOPFCase(data_path="ieee123_30der")
    >>> # Initialize the LinDistModel
    >>> model = LinDistModel(
    ...     branch_data=case.branch_data,
    ...     bus_data=case.bus_data,
    ...     gen_data=case.gen_data,
    ...     cap_data=case.cap_data,
    ...     reg_data=case.reg_data,
    ... )
    >>> # Solve the model using the specified objective function
    >>> result = opf.lp_solve(model, opf.gradient_load_min(model))
    >>> # Extract and plot results
    >>> v = model.get_voltages(result.x)
    >>> s = model.get_apparent_power_flows(result.x)
    >>> p_gens = model.get_p_gens(result.x)
    >>> q_gens = model.get_q_gens(result.x)
    >>> # Visualize network and power flows
    >>> opf.plot_network(model, v=v, s=s, p_gen=p_gens, q_gen=q_gens).show()
    >>> opf.plot_voltages(v).show()
    >>> opf.plot_power_flows(s).show()
    >>> opf.plot_gens(p_gens, q_gens).show()
    """

    def initialize_variable_index_pointers(self):
        self.x_maps, self.n_x = self._variable_tables(self.branch)
        self.v_map, self.n_x = self._add_device_variables(self.n_x, self.all_buses)
        self.pg_map, self.n_x = self._add_device_variables(self.n_x, self.gen_buses)
        self.qg_map, self.n_x = self._add_device_variables(self.n_x, self.gen_buses)
        self.qc_map, self.n_x = self._add_device_variables(self.n_x, self.cap_buses)
        self.vx_map, self.n_x = self._add_device_variables(self.n_x, self.reg_buses)

    def build(self):
        self.initialize_variable_index_pointers()
        self.a_eq, self.b_eq = self.create_model()
        self.a_ub, self.b_ub = self.create_inequality_constraints()
        self.bounds = self.init_bounds()
        self.bounds_tuple = list(map(tuple, self.bounds))
        self.x_min = self.bounds[:, 0]
        self.x_max = self.bounds[:, 1]

    @staticmethod
    def _variable_tables(branch, n_x=0) -> Tuple[dict[str, pd.DataFrame], int]:
        """
        Constructs tables to map branch power variables to indices within the
        optimization variable vector.

        Parameters
        ----------
        branch : pd.DataFrame
            DataFrame containing branch information.
        n_x : int
            Initial variable index; used to offset additional device variables.

        Returns
        -------
        x_maps : dict
            Dictionary of DataFrames mapping branch indices to variable indices.
        n_x : int
            Updated variable index after accounting for new variables.
        """
        x_maps = {}
        for a in "abc":
            indices = branch.phases.str.contains(a)
            lines = branch.loc[indices, ["fb", "tb"]].values.astype(int) - 1
            n_lines = len(lines)
            df = pd.DataFrame(columns=["bi", "bj", "pij", "qij"], index=range(n_lines))
            if n_lines == 0:
                x_maps[a] = df.astype(int)
                continue
            g: nx.Graph = nx.Graph()
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
    def _add_device_variables(
        n_x: int, device_buses: dict
    ) -> Tuple[dict[str, pd.Series], int]:
        """
        Adds device-related variables (e.g., voltage, power load) to the
        optimization problem, indexed by the phase.

        Parameters
        ----------
        n_x : int
            Starting offset for variable indices.
        device_buses : dict
            Dictionary containing bus indices for devices, categorized by phase.

        Returns
        -------
        device_maps : dict
            Mapping of variable indices to bus indices for each device.
        n_x : int
            Updated offset after adding device variables.
        """
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

    def init_bounds(self) -> np.ndarray:
        """
        Initializes the variable bounds for the optimization problem.

        Returns
        -------
        bounds : np.ndarray
            Array of upper and lower bounds for each variable.
        """
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

    def user_added_limits(
        self, x_lim_lower, x_lim_upper
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def add_voltage_limits(
        self, x_lim_lower, x_lim_upper
    ) -> Tuple[np.ndarray, np.ndarray]:
        for a in "abc":
            if not self.phase_exists(a):
                continue
            # ~~ v limits ~~:
            x_lim_upper[self.v_map[a]] = self.bus.loc[self.v_map[a].index, "v_max"] ** 2
            x_lim_lower[self.v_map[a]] = self.bus.loc[self.v_map[a].index, "v_min"] ** 2
        return x_lim_lower, x_lim_upper

    def add_generator_limits(
        self, x_lim_lower, x_lim_upper
    ) -> Tuple[np.ndarray, np.ndarray]:
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
                mode = self.gen.loc[j, "control_variable"]
                pg = self.idx("pg", j, a)
                qg = self.idx("qg", j, a)
                # active power bounds
                x_lim_lower[pg] = 0
                x_lim_upper[pg] = p_out[j]
                # reactive power bounds
                if mode == opf.CONSTANT_P:
                    x_lim_lower[qg] = max(q_min[j], q_min_manual[j])
                    x_lim_upper[qg] = min(q_max[j], q_max_manual[j])
                if mode != opf.CONSTANT_P:
                    # reactive power bounds
                    x_lim_lower[qg] = max(-s_rated[j], q_min_manual[j])
                    x_lim_upper[qg] = min(s_rated[j], q_max_manual[j])
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
        if var in ["pg", "p_gen"]:  # active power generation at node
            return self.pg_map[phase].get(node_j, [])
        if var in ["qg", "q_gen"]:  # reactive power generation at node
            return self.qg_map[phase].get(node_j, [])
        if var in ["qc", "q_cap"]:  # reactive power injection by capacitor
            return self.qc_map[phase].get(node_j, [])
        if var in ["vx"]:
            return self.vx_map[phase].get(node_j, [])
        ix = self.additional_variable_idx(var, node_j, phase)
        if ix is not None:
            return ix
        raise ValueError(f"Variable name, '{var}', not found.")

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
        return None

    @cache
    def phase_exists(self, phase, index: Optional[int] = None) -> bool:
        if index is None:
            return self.x_maps[phase].shape[0] > 0
        return len(self.idx("bj", index, phase)) > 0

    def create_model(self) -> Tuple[csr_array, np.ndarray]:
        """
        Constructs the equality constraint matrices for the linear optimization
        problem based on power flow equations.
        a_eq*x == b_eq

        Returns
        -------
        a_eq : csr_array
            Sparse array representing the equality constraint matrix.
        b_eq : np.ndarray
            Array representing the equality constraint vector.
        """
        # ########## Aeq and Beq Formation ###########
        n_rows = self.n_x
        n_cols = self.n_x
        # Aeq has the same number of rows as equations with a column for each x
        a_eq = lil_array((n_rows, n_cols))
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

    def add_power_flow_model(
        self, a_eq: lil_array, b_eq, j, phase
    ) -> Tuple[lil_array, np.ndarray]:
        pij = self.idx("pij", j, phase)
        qij = self.idx("qij", j, phase)
        pjk = self.idx("pjk", j, phase)
        qjk = self.idx("qjk", j, phase)
        # Set P equation variable coefficients in a_eq
        a_eq[pij, pij] = 1
        a_eq[pij, pjk] = -1
        # Set Q equation variable coefficients in a_eq
        a_eq[qij, qij] = 1
        a_eq[qij, qjk] = -1
        return a_eq, b_eq

    def add_voltage_drop_model(
        self, a_eq: lil_array, b_eq, j, a, b, c
    ) -> Tuple[lil_array, np.ndarray]:
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

    def add_regulator_model(
        self, a_eq: lil_array, b_eq, j, a
    ) -> Tuple[lil_array, np.ndarray]:
        if self.reg is None:
            return a_eq, b_eq
        if j not in self.reg.tb:
            return a_eq, b_eq
        i = self.idx("bi", j, a)[0]  # get the upstream node, i, on branch from i to j
        pij = self.idx("pij", j, a)
        qij = self.idx("qij", j, a)
        vi = self.idx("v", i, a)
        vj = self.idx("v", j, a)
        vx = self.idx("vx", j, a)
        r, x = self.r, self.x
        aa = "".join(sorted(a + a))

        a_eq[vj, vj] = 1
        a_eq[vj, vx] = -1
        a_eq[vj, pij] = 2 * r[aa][i, j]
        a_eq[vj, qij] = 2 * x[aa][i, j]

        reg_ratio = get(self.reg[f"ratio_{a}"], j, 1)
        a_eq[vx, vx] = 1
        a_eq[vx, vi] = -1 * reg_ratio**2
        return a_eq, b_eq

    def add_swing_voltage_model(
        self, a_eq: lil_array, b_eq, j, a
    ) -> Tuple[lil_array, np.ndarray]:
        i = self.idx("bi", j, a)[0]  # get the upstream node, i, on branch from i to j
        vi = self.idx("v", i, a)
        # Set V equation variable coefficients in a_eq and constants in b_eq
        if self.bus.bus_type[i] == opf.SWING_BUS:  # Swing bus
            a_eq[vi, vi] = 1
            b_eq[vi] = self.bus.at[i, f"v_{a}"] ** 2
        return a_eq, b_eq

    def add_generator_model(
        self, a_eq: lil_array, b_eq, j, a
    ) -> Tuple[lil_array, np.ndarray]:
        p_gen_nom, q_gen_nom = 0, 0
        if self.gen is not None:
            p_gen_nom = get(self.gen[f"p{a}"], j, 0)
            q_gen_nom = get(self.gen[f"q{a}"], j, 0)
        # equation indexes
        pij = self.idx("pij", j, a)
        qij = self.idx("qij", j, a)
        pg = self.idx("pg", j, a)
        qg = self.idx("qg", j, a)
        # Set Generator equation variable coefficients in a_eq
        a_eq[pij, pg] = 1
        a_eq[qij, qg] = 1
        if get(self.gen["control_variable"], j, 0) in [opf.CONSTANT_PQ, opf.CONSTANT_P]:
            a_eq[pg, pg] = 1
            b_eq[pg] = p_gen_nom
        if get(self.gen["control_variable"], j, 0) in [opf.CONSTANT_PQ, opf.CONSTANT_Q]:
            a_eq[qg, qg] = 1
            b_eq[qg] = q_gen_nom
        return a_eq, b_eq

    def add_load_model(
        self, a_eq: lil_array, b_eq, j, a
    ) -> Tuple[lil_array, np.ndarray]:
        pij = self.idx("pij", j, a)
        qij = self.idx("qij", j, a)
        vj = self.idx("v", j, a)
        p_load_nom, q_load_nom = 0, 0
        if self.bus.bus_type[j] == opf.PQ_BUS:
            p_load_nom = self.bus[f"pl_{a}"][j]
            q_load_nom = self.bus[f"ql_{a}"][j]
        if self.bus.bus_type[j] != opf.PQ_FREE:
            # Set loads model to power flow equation
            a_eq[pij, vj] = -(self.bus.cvr_p[j] / 2) * p_load_nom
            b_eq[pij] = (1 - (self.bus.cvr_p[j] / 2)) * p_load_nom
            a_eq[qij, vj] = -(self.bus.cvr_q[j] / 2) * q_load_nom
            b_eq[qij] = (1 - (self.bus.cvr_q[j] / 2)) * q_load_nom
        return a_eq, b_eq

    def add_capacitor_model(
        self, a_eq: lil_array, b_eq, j, a
    ) -> Tuple[lil_array, np.ndarray]:
        q_cap_nom = 0
        if self.cap is not None:
            q_cap_nom = get(self.cap[f"q{a}"], j, 0)
        # equation indexes
        qij = self.idx("qij", j, a)
        vj = self.idx("v", j, a)
        qc = self.idx("q_cap", j, a)
        a_eq[qij, qc] = 1  # add capacitor q variable to power flow equation
        a_eq[qc, qc] = 1
        a_eq[qc, vj] = -q_cap_nom
        return a_eq, b_eq

    def create_inequality_constraints(self) -> Tuple[csr_array, np.ndarray]:
        """
        Constructs the inequality constraint matrices.
        a_ub*x <= b_ub

        Returns
        -------
        a_ub, : csr_array
            Sparse array representing the inequality constraint matrix.
        b_ub : np.ndarray
            Array representing the inequality constraint vector.
        """
        a_ub, b_ub = self.create_octagon_constraints()
        return csr_array(a_ub), b_ub

    def create_hexagon_constraints(self) -> Tuple[csr_array, np.ndarray]:
        """
        Use a hexagon to approximate the circular inequality constraint of an inverter.
        """
        n_inequalities = 5
        n_rows_ineq = n_inequalities * (
            len(np.where(self.gen.control_variable == opf.CONTROL_PQ)[0]) * 3
        )
        n_rows_ineq = max(n_rows_ineq, 1)
        a_ineq = lil_array((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq = list(range(n_inequalities))  # initialize equation indices
        for j in self.gen.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue
                if self.gen.loc[j, "control_variable"] != opf.CONTROL_PQ:
                    continue
                pg = self.idx("pg", j, a)
                qg = self.idx("qg", j, a)
                s_rated = self.gen.at[j, f"s{a}_max"]
                coef = sqrt(3) / 3  # ~=0.5774
                # Right half plane. Positive P
                # limit for small +P and large +Q
                a_ineq[ineq[0], pg] = 0
                a_ineq[ineq[0], qg] = 2 * coef
                b_ineq[ineq[0]] = s_rated
                # limit for large +P and small +Q
                a_ineq[ineq[1], pg] = 1
                a_ineq[ineq[1], qg] = coef
                b_ineq[ineq[1]] = s_rated
                # limit for large +P and small -Q
                a_ineq[ineq[2], pg] = 1
                a_ineq[ineq[2], qg] = -coef
                b_ineq[ineq[2]] = s_rated
                # limit for small +P and large -Q
                a_ineq[ineq[3], pg] = 0
                a_ineq[ineq[3], qg] = -2 * coef
                b_ineq[ineq[3]] = s_rated
                # limit to right half plane
                a_ineq[ineq[4], pg] = -1
                b_ineq[ineq[4]] = 0
                # increment equation indices
                for n_ineq in range(len(ineq)):
                    ineq[n_ineq] += len(ineq)
        return csr_array(a_ineq), b_ineq

    def create_octagon_constraints(self) -> Tuple[csr_array, np.ndarray]:
        """
        Use an octagon to approximate the circular inequality constraint of an inverter.
        """

        # ########## Aineq and Bineq Formation ###########
        n_inequalities = 5

        n_rows_ineq = n_inequalities * (
            len(np.where(self.gen.control_variable == opf.CONTROL_PQ)[0]) * 3
        )
        n_rows_ineq = max(n_rows_ineq, 1)
        a_ineq = lil_array((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq = list(range(n_inequalities))
        for j in self.gen.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue
                if self.gen.loc[j, "control_variable"] != opf.CONTROL_PQ:
                    continue
                pg = self.idx("pg", j, a)
                qg = self.idx("qg", j, a)
                s_rated = self.gen.at[j, f"s{a}_max"]
                coef = sqrt(2) - 1  # ~=0.4142
                # Right half plane. Positive P
                # limit for small +P and large +Q
                a_ineq[ineq[0], pg] = coef
                a_ineq[ineq[0], qg] = 1
                b_ineq[ineq[0]] = s_rated
                # limit for large +P and small +Q
                a_ineq[ineq[1], pg] = 1
                a_ineq[ineq[1], qg] = coef
                b_ineq[ineq[1]] = s_rated
                # limit for large +P and small -Q
                a_ineq[ineq[2], pg] = 1
                a_ineq[ineq[2], qg] = -coef
                b_ineq[ineq[2]] = s_rated
                # limit for small +P and large -Q
                a_ineq[ineq[3], pg] = coef
                a_ineq[ineq[3], qg] = -1
                b_ineq[ineq[3]] = s_rated
                # limit to right half plane
                a_ineq[ineq[4], pg] = -1
                b_ineq[ineq[4]] = 0

                for n_ineq in range(len(ineq)):
                    ineq[n_ineq] += len(ineq)
        return csr_array(a_ineq), b_ineq

    def parse_results(self, x, variable_name: str):
        values = pd.DataFrame(columns=["name", "a", "b", "c"])
        for ph in "abc":
            for j in self.all_buses[ph]:
                values.at[j + 1, "name"] = self.bus.at[j, "name"]
                values.at[j + 1, ph] = x[self.idx(variable_name, j, ph)]
        return values.sort_index()

    def get_device_variables(self, x, variable_map):
        if len(variable_map.keys()) == 0:
            return pd.DataFrame(columns=["id", "name", "t", "a", "b", "c"])
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

    def get_p_gens(self, x):
        return self.get_device_variables(x, self.pg_map)

    def get_q_gens(self, x):
        return self.get_device_variables(x, self.qg_map)

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

    def update(
        self,
        bus_data: Optional[pd.DataFrame] = None,
        gen_data: Optional[pd.DataFrame] = None,
        cap_data: Optional[pd.DataFrame] = None,
        reg_data: Optional[pd.DataFrame] = None,
    ):

        # TODO: update is untested!
        warnings.warn("update method is untested!")
        if bus_data is not None:
            self.bus = handle_bus_input(bus_data)
        if gen_data is not None:
            self.gen = handle_gen_input(gen_data)
        if cap_data is not None:
            self.cap = handle_cap_input(cap_data)
        if reg_data is not None:
            self.reg = handle_reg_input(reg_data)

        a_eq = lil_array(self.a_eq)
        for j in range(1, self.nb):
            for ph in ["abc", "bca", "cab"]:
                a, b, c = ph[0], ph[1], ph[2]
                if not self.phase_exists(a, j):
                    continue
                if bus_data is not None:
                    a_eq, self.b_eq = self.add_swing_voltage_model(
                        a_eq, self.b_eq, j, a
                    )
                    a_eq, self.b_eq = self.add_load_model(a_eq, self.b_eq, j, a)
                if gen_data is not None:
                    a_eq, self.b_eq = self.add_generator_model(a_eq, self.b_eq, j, a)
                if cap_data is not None:
                    a_eq, self.b_eq = self.add_capacitor_model(a_eq, self.b_eq, j, a)
                if reg_data is not None:
                    a_eq, self.b_eq = self.add_regulator_model(a_eq, self.b_eq, j, a)
        self.a_eq = csr_array(a_eq)
        self.a_ub, self.b_ub = self.create_inequality_constraints()
        self.bounds = self.init_bounds()
        self.bounds_tuple = list(map(tuple, self.bounds))
        self.x_min = self.bounds[:, 0]
        self.x_max = self.bounds[:, 1]
