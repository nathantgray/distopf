from typing import Optional
import pandas as pd
import distopf as opf
from distopf.multiperiod.base_mp import LinDistBaseMP


class LinDistMPL(LinDistBaseMP):
    """
    LinDistFlow Model for multiperiod linear power flow modeling which includes active and
    reactive load powers as variables. This may be useful starting point if
    custom load models need to be added. The disadvantage is that significantly
    more variables are included which will increase computation time.

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
    """

    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
        bat_data: pd.DataFrame = None,
        loadshape_data: pd.DataFrame = None,
        pv_loadshape_data: pd.DataFrame = None,
        start_step: int = 0,
        n_steps: int = 24,
        delta_t: float = 1,  # hours per step
    ):
        super().__init__(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            cap_data=cap_data,
            reg_data=reg_data,
            bat_data=bat_data,
            loadshape_data=loadshape_data,
            pv_loadshape_data=pv_loadshape_data,
            start_step=start_step,
            n_steps=n_steps,
            delta_t=delta_t,
        )
        self.pl_map: Optional[dict[int, dict[str, pd.Series]]] = None
        self.ql_map: Optional[dict[int, dict[str, pd.Series]]] = None
        self.build()

    def initialize_variable_index_pointers(self):
        # ~~ initialize index pointers ~~
        self.x_maps = {}
        self.v_map = {}
        self.pg_map = {}
        self.qg_map = {}
        self.qc_map = {}
        self.charge_map = {}
        self.discharge_map = {}
        self.soc_map = {}
        self.vx_map = {}
        self.pl_map = {}
        self.ql_map = {}
        self.n_x = 0
        for t in range(self.start_step, self.start_step + self.n_steps):
            self.x_maps[t], self.n_x = self._variable_tables(self.branch, n_x=self.n_x)
            self.v_map[t], self.n_x = self._add_device_variables(
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
            self.vx_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.reg_buses
            )
            self.pl_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.load_buses
            )
            self.ql_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.load_buses
            )

    def additional_variable_idx(self, var, node_j, phase, t=0):
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
        if t < self.start_step:
            t = self.start_step
        if var in ["pl", "p_load"]:  # reactive power load at node
            return self.pl_map[t][phase].get(node_j, [])
        if var in ["ql", "q_load"]:  # reactive power injection by capacitor
            return self.ql_map[t][phase].get(node_j, [])
        return None

    def add_load_model(self, a_eq, b_eq, j, a, t=0):
        pij = self.idx("pij", j, a, t=t)
        qij = self.idx("qij", j, a, t=t)
        pl = self.idx("pl", j, a, t=t)
        ql = self.idx("ql", j, a, t=t)
        vj = self.idx("v", j, a, t=t)
        p_load_nom, q_load_nom = 0, 0
        load_mult = self.loadshape.M[t]
        a_eq[pij, pl] = -1  # add load variable to power flow equation
        a_eq[qij, ql] = -1  # add load variable to power flow equation
        if self.bus.bus_type[j] == opf.PQ_BUS:
            p_load_nom = self.bus[f"pl_{a}"][j] * load_mult
            q_load_nom = self.bus[f"ql_{a}"][j] * load_mult
        if self.bus.bus_type[j] != opf.PQ_FREE:
            # Set Load equation variable coefficients in a_eq
            a_eq[pl, pl] = 1
            a_eq[pl, vj] = -(self.bus.cvr_p[j] / 2) * p_load_nom
            b_eq[pl] = (1 - (self.bus.cvr_p[j] / 2)) * p_load_nom

            a_eq[ql, ql] = 1
            a_eq[ql, vj] = -(self.bus.cvr_q[j] / 2) * q_load_nom
            b_eq[ql] = (1 - (self.bus.cvr_q[j] / 2)) * q_load_nom
        return a_eq, b_eq

    def get_p_loads(self, x):
        return self.get_device_variables(x, self.pl_map)

    def get_q_loads(self, x):
        return self.get_device_variables(x, self.ql_map)
