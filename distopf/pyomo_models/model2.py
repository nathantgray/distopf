import numpy as np
import pandas as pd
from scipy.sparse import csr_array, lil_array, vstack  # type: ignore
from distopf.utils import (
    handle_branch_input,
    handle_bus_input,
    handle_gen_input,
    handle_cap_input,
    handle_reg_input,
    # handle_bat_input,
    handle_loadshape_input,
    handle_pv_loadshape_input,
    get,
)
from distopf.plot import plot_voltages_timeseries, plot_power_flows_timeseries, plot_gens_timeseries, plot_device_timeseries
from distopf.pyomo_models.objective import loss_minimize_with_scd
from distopf.pyomo_models.constraints import (
    include_substation_voltage_constraint,
    include_voltage_drop_constraint,
    include_voltage_limit_constraint,
    include_soc_limits,
    include_p_gen_constraint,
    include_q_gen_limit,
    include_final_soc_constraint,
    include_charge_state_constraint,
    include_bat_power_limits,
    include_lindist_p_flow_constraint,
    include_lindist_q_flow_constraint,
)
import math
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Param,
    Set,
    NonNegativeReals, # type: ignore
    minimize,
    sqrt,
    inequality,
    Binary, # type: ignore
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    Objective,
)
from pathlib import Path
from distopf.cases import CASES_DIR


def handle_bat_input(bat_data: pd.DataFrame) -> pd.DataFrame:
    if bat_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "s_max",
                "phases",
                "energy_capacity",
                "min_soc",
                "max_soc",
                "start_soc",
                "charge_efficiency",
                "discharge_efficiency",
            ]
        )
    bat = bat_data.sort_values(by="id", ignore_index=True)
    bat.index = bat.id.to_numpy() - 1
    return bat


class PyoMP2:
    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
        bat_data: pd.DataFrame = None,
        schedules: pd.DataFrame = None,
        start_step: int = 0,
        n_steps: int = 24,
        delta_t: float = 1,  # hours per step
    ):
        self.cm = ConcreteModel(name="BaseModelMP")
        # ~~~~~~~~~~~~~~~~~~~~ Load Data Frames ~~~~~~~~~~~~~~~~~~~~
        self.branch = handle_branch_input(branch_data)
        self.bus = handle_bus_input(bus_data)
        self.gen = handle_gen_input(gen_data)
        self.cap = handle_cap_input(cap_data)
        self.reg = handle_reg_input(reg_data)
        self.bat = handle_bat_input(bat_data)
        self.schedules = handle_loadshape_input(schedules)
        self.start_step = start_step
        self.n_steps = n_steps
        self.delta_t = delta_t  # hours per step

        self.nb = len(self.bus.id)
        self.r, self.x = self._init_rx()
        self.swing_bus_id = self.bus.loc[self.bus.bus_type == "SWING", "id"].tolist()[0]

        self.t_set = np.arange(start_step, start_step + n_steps)
        self.cm = ConcreteModel(name="BaseModelMP")
        self.v_swing = {}
        self.v_max = {}
        self.v_min = {}
        self.p_load = {}
        self.q_load = {}
        self.s_gen = {}
        self.p_gen = {}
        self.q_gen = {}
        self.p_bat = {}
        self.q_bat = {}
        self.p_bat_in = {}
        self.p_bat_out = {}

    def _init_rx(self):
        branch = self.branch
        row = np.array(np.r_[branch.fb, branch.tb], dtype=int)
        col = np.array(np.r_[branch.tb, branch.fb], dtype=int)
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

    def _parse_bus_data(self):
        bus_lookup = self.bus.set_index("id")
        p_load_shape = {}
        q_load_shape = {}
        for i in self.bus["id"]:
            for ph in "abc":
                load_shape = bus_lookup.at[i, f"load_shape"]
                if load_shape in self.schedules.columns:
                    p_load_shape[(i, ph)] = self.schedules[load_shape]
                    q_load_shape[(i, ph)] = self.schedules[load_shape]
                elif f"{load_shape}.{ph}.p" in self.schedules.columns:
                    p_load_shape[(i, ph)] = self.schedules[f"{load_shape}.{ph}.p"]
                    q_load_shape[(i, ph)] = self.schedules[f"{load_shape}.{ph}.q"]
        self.p_load = {
            (t, i, ph): bus_lookup.at[i, f"pl_{ph}"] * p_load_shape[(i, ph)][t]
            for t in self.t_set
            for i in self.bus["id"]
            for ph in "abc"
        }
        self.q_load = {
            (t, i, ph): bus_lookup.at[i, f"ql_{ph}"] * q_load_shape[(i, ph)][t]
            for t in self.t_set
            for i in self.bus["id"]
            for ph in "abc"
        }
        self.v_min = bus_lookup.v_min.to_dict()
        self.v_max = bus_lookup.v_max.to_dict()
        self.v_swing = {
            (t, i, ph): bus_lookup.at[i, f"v_{ph}"]
            for t in self.t_set
            for i in [self.swing_bus_id]
            for ph in "abc"
        }

    def import_regulator_data(self):
        self.cm.tap = {}
        self.cm.ratio = {}
        if self.reg is None:
            return
        reg_lookup = self.reg.set_index("tb")
        self.cm.tap = {
            (j, ph): get(reg_lookup[f"tap_{ph}"], j, 0)
            for j in self.reg["tb"]
            for ph in "abc"
        }
        self.cm.ratio = {
            (j, ph): get(reg_lookup[f"ratio_{ph}"], j, )
            for j in self.reg["tb"]
            for ph in "abc"
        }

    def import_generator_data(self):
        ## parsing gen_data
        self.cm.p_gen_setpoint = {}
        self.cm.q_gen_setpoint = {}
        self.cm.s_gen_rated = {}
        self.cm.control_variable = {}
        if self.gen is None:
            return 
        gen_lookup = self.gen.set_index("id")
        self.cm.p_gen_setpoint = {
            (t, i, ph): gen_lookup.at[i, f"p{ph}"] * self.schedules.PV[t]
            for t in self.t_set
            for i in self.gen["id"]
            for ph in "abc"
        }
        self.cm.q_gen_setpoint = {
            (t, i, ph): gen_lookup.at[i, f"q{ph}"]
            for t in self.t_set
            for i in self.gen["id"]
            for ph in "abc"
        }
        self.cm.s_gen_rated = {
            (i, ph): gen_lookup.at[i, f"s{ph}_max"]
            for i in self.gen["id"]
            for ph in "abc"
        }
        self.cm.q_max_manual = {
            (i, ph): gen_lookup.get(f"q{ph}_max", np.ones_like(q_min) * 100e3)
            for i in self.gen["id"]
            for ph in "abc"
        }
        self.cm.q_min_manual = {
            (i, ph): gen_lookup.get(f"q{ph}_min", np.ones_like(q_min) * -100e3)
            for i in self.gen["id"]
            for ph in "abc"
        }
        self.cm.control_variable = {
            (i, ph): gen_lookup.at[i, f"control_variable"]
            for i in self.gen["id"]
            for ph in "abc"
        }

    def import_battery_data_old(self):
        ## parsing bat_data
        self.cm.p_bat_max = {}
        self.cm.s_bat_max = {}
        self.cm.charge_efficiency = {}
        self.cm.discharge_efficiency = {}
        self.cm.charge_state_min = {}
        self.cm.charge_state_max = {}
        self.cm.charge_state_initial = {}
        if self.bat is None:
            return 
        bat_lookup = self.bat.set_index("id")
        self.cm.p_bat_max = {
            (i, ph): bat_lookup.at[i, f"Pb_max_{ph}"]
            for i in self.bat["id"]
            for ph in "abc"
        }
        self.cm.s_bat_max = {
            (i, ph): bat_lookup.at[i, f"hmax_{ph}"]
            for i in self.bat["id"]
            for ph in "abc"
        }
        self.cm.charge_efficiency = {
            (i, ph): bat_lookup.at[i, f"nc_{ph}"]
            for i in self.bat["id"]
            for ph in "abc"
        }
        self.cm.discharge_efficiency = {
            (i, ph): bat_lookup.at[i, f"nd_{ph}"]
            for i in self.bat["id"]
            for ph in "abc"
        }
        self.cm.charge_state_min = {
            (i, ph): bat_lookup.at[i, f"bmin_{ph}"]
            for i in self.bat["id"]
            for ph in "abc"
        }
        self.cm.charge_state_max = {
            (i, ph): bat_lookup.at[i, f"bmax_{ph}"]
            for i in self.bat["id"]
            for ph in "abc"
        }
        self.cm.charge_state_initial = {
            (i, ph): (
                self.cm.charge_state_min[(i, ph)]
                + self.cm.charge_state_max[(i, ph)]
            )
            / 2
            for i in self.bat["id"]
            for ph in "abc"
        }

    def import_bus_data(self):
        self._parse_bus_data()
        self.cm.p_load = Param(
            self.cm.t_set,
            self.cm.node_set,
            self.cm.phase_set,
            initialize=self.p_load,
            mutable=True,
        )
        self.cm.q_load = Param(
            self.cm.t_set,
            self.cm.node_set,
            self.cm.phase_set,
            initialize=self.q_load,
            mutable=True,
        )
        self.cm.v_swing = Param(
            self.cm.t_set,
            self.cm.swing_bus_set,
            self.cm.phase_set,
            initialize=self.v_swing,
            mutable=True,
        )
        self.cm.v_max = Param(self.cm.node_set, initialize=self.v_max, mutable=True)
        self.cm.v_min = Param(self.cm.node_set, initialize=self.v_min, mutable=True)

    # def import_generator_parameters(self):

    def build(self):
        # Sets
        self.cm.t_set = self.t_set
        self.cm.node_set = sorted(set(self.bus["id"]))
        self.cm.link_set = sorted(set(zip(self.branch["fb"], self.branch["tb"])))
        self.cm.gen_set = sorted(set(self.gen["id"]))
        self.cm.cap_set = sorted(set(self.cap["id"]))
        self.cm.reg_set = sorted(set(self.reg["tb"]))
        self.cm.v_reg_set = sorted(set(zip(self.reg["fb"], self.reg["tb"])))
        self.cm.bat_set = sorted(set(self.bat["id"]))
        self.cm.phase_set = sorted({"a", "b", "c"})
        self.cm.swing_bus_set = sorted({self.swing_bus_id})

        ## initializing model parameters
        self.cm.n_steps = self.n_steps
        self.cm.swing_bus = self.swing_bus_id
        self.cm.r = self.r
        self.cm.x = self.x
        self.cm.cost = self.schedules.price

        self.import_bus_data()
        self.import_regulator_data()
        self.import_generator_data()
        self.import_battery_data_old()

        # Variables
        self.cm.p_subs = Var(self.cm.t_set, self.cm.phase_set, domain=NonNegativeReals)
        self.cm.q_subs = Var(self.cm.t_set, self.cm.phase_set, domain=NonNegativeReals)
        self.cm.p = Var(self.cm.t_set, self.cm.link_set, self.cm.phase_set)
        self.cm.q = Var(self.cm.t_set, self.cm.link_set, self.cm.phase_set)
        self.cm.v = Var(
            self.cm.t_set, self.cm.node_set, self.cm.phase_set, domain=NonNegativeReals
        )
        self.cm.p_gen = Var(self.cm.t_set, self.cm.gen_set, self.cm.phase_set)
        self.cm.q_gen = Var(self.cm.t_set, self.cm.gen_set, self.cm.phase_set)
        self.cm.q_cap = Var(self.cm.t_set, self.cm.cap_set, self.cm.phase_set)
        self.cm.v_reg = Var(self.cm.t_set, self.cm.reg_set, self.cm.phase_set)
        self.cm.p_charge = Var(
            self.cm.t_set, self.cm.bat_set, self.cm.phase_set, domain=NonNegativeReals
        )
        self.cm.p_discharge = Var(
            self.cm.t_set, self.cm.bat_set, self.cm.phase_set, domain=NonNegativeReals
        )
        self.cm.p_batt = Var(
            self.cm.t_set, self.cm.bat_set, self.cm.phase_set, domain=NonNegativeReals
        )
        self.cm.charge_state = Var(
            self.cm.t_set, self.cm.bat_set, self.cm.phase_set, domain=NonNegativeReals
        )
        # constraints

        include_net_batt_power_equation(self.cm)
        include_substation_voltage_constraint(self.cm)
        include_lindist_p_flow_constraint(self.cm)
        include_lindist_q_flow_constraint(self.cm)
        include_voltage_drop_constraint(self.cm)
        include_voltage_limit_constraint(self.cm)
        include_p_gen_constraint(self.cm)
        include_q_gen_limit(self.cm)
        include_bat_power_limits(self.cm)
        include_final_soc_constraint(self.cm)
        include_soc_limits(self.cm)
        include_charge_state_constraint(self.cm)

    def solve(self, obj_func, **kwargs):
        # Store kwargs as attributes on the model
        for key, value in kwargs.items():
            setattr(self.cm, key, value)
        if hasattr(self.cm, "obj"):
            self.cm.del_component("obj")  # Remove old objective

        self.cm.obj = Objective(rule=obj_func, sense=minimize)
        opt = SolverFactory("ipopt")
        results = opt.solve(self.cm, tee=False)
        if (
            results.solver.status == "ok"
            and results.solver.termination_condition == "optimal"
        ):
            print("Solver completed successfully.")
        else:
            print(f"Solver failed: {results.solver.termination_condition}")

        return self.cm

    def get_voltages_tidy(self):
        v_dict = self.cm.v.extract_values()
        v = [
            [t, j, ph, (v_dict[t, j, ph]) ** (1 / 2)]
            for t in self.cm.t_set
            for j in self.cm.node_set
            for ph in "abc"
        ]
        v_df = pd.DataFrame(columns=["t", "id", "phase", "value"], data=v)
        return v_df
    
    def get_voltages(self):
        v = extract_nodal_values_tidy(self.cm.v)
        v.value = v.value**(1/2)
        v = v.pivot(
            index=["t", "id"], columns="phase", values="value"
        ).reset_index()
        v.columns.name = None
        return v

    def get_p_gens(self):
        return extract_nodal_values_df(self.cm.p_gen)

    def get_q_gens(self):
        return extract_nodal_values_df(self.cm.q_gen)

    # def get_p_batt(self):
    #     return extract_nodal_values_df(self.cm.p_bat)

    # def get_q_batt(self):
    #     return extract_nodal_values_df(self.cm.q_batt)

    # def get_q_caps(self):
    #     return extract_nodal_values_df(self.cm.q_cap)

    def get_p_charge(self):
        return extract_nodal_values_df(self.cm.p_charge)

    def get_p_discharge(self):
        return extract_nodal_values_df(self.cm.p_discharge)

    def get_soc(self):
        return extract_nodal_values_df(self.cm.charge_state)

    def get_active_power_flows(self):
        return extract_link_values_df(self.cm.p)

    def get_reactive_power_flows(self):
        return extract_link_values_df(self.cm.q)
    
    def get_apparent_power_flows(self):
        df = extract_link_values_tidy(self.cm.p)
        dfq = extract_link_values_tidy(self.cm.q)
        df.value = df.value + 1j * dfq.value
        df = df.pivot(
            index=["t", "fb", "tb"], columns="phase", values="value"
        ).reset_index()
        df.columns.name = None
        return df


def extract_nodal_values_tidy(var: Var):
    data = var.extract_values()
    df = pd.DataFrame(
        [(*key, value) for key, value in data.items()],
        columns=["t", "id", "phase", "value"],
    )
    return df


def extract_link_values_tidy(var: Var):
    data = var.extract_values()
    df = pd.DataFrame(
        [(*key, value) for key, value in data.items()],
        columns=["t", "fb", "tb", "phase", "value"],
    )
    return df


def extract_nodal_values_df(var: Var):
    df_intermediate = extract_nodal_values_tidy(var)
    df = df_intermediate.pivot(
        index=["t", "id"], columns="phase", values="value"
    ).reset_index()
    df.columns.name = None
    return df


def extract_link_values_df(var: Var):
    df_intermediate = extract_link_values_tidy(var)
    df = df_intermediate.pivot(
        index=["t", "fb", "tb"], columns="phase", values="value"
    ).reset_index()
    df.columns.name = None
    return df


def include_net_batt_power_equation(cm: ConcreteModel):
    def p_batt_rule(m: ConcreteModel, t, j, ph):
        return m.p_gen[t, j, ph] == m.p_gen_setpoint[t, j, ph]
    cm.p_batt_constraint = Constraint(cm.t_set, cm.bat_set, cm.phase_set, rule=p_batt_rule)




def include_voltage_drop_constraint(cm: ConcreteModel):
    cm.kvl_three_phase = Constraint(
        cm.t_set, cm.link_set, cm.phase_set, rule=kvl_three_phase_rule
    )

def kvl_three_phase_rule(m: ConcreteModel, t, i, j, ph):
    if (i, j) in m.reg_set:
        return Constraint.Skip
    a, b, c = "a", "b", "c"
    aa, ab, ac = "aa", "ab", "ac"
    if ph == "b":
        a, b, c = "b", "c", "a"
        aa, ab, ac = "bb", "bc", "ab"
    if ph == "c":
        a, b, c = "c", "a", "b"
        aa, ab, ac = "cc", "ac", "bc"
    return (
        m.v[t, j, a]
        - m.v[t, i, a]
        + 2 * (m.r[aa][i, j] * m.p[t, i, j, a] + m.x[aa][i, j] * m.q[t, i, j, a])
        + (-m.r[ab][i, j] + sqrt(3) * m.x[ab][i, j]) * m.p[t, i, j, b]
        + (-m.x[ab][i, j] - sqrt(3) * m.r[ab][i, j]) * m.q[t, i, j, b]
        + (-m.r[ac][i, j] - sqrt(3) * m.x[ac][i, j]) * m.p[t, i, j, c]
        + (-m.x[ac][i, j] + sqrt(3) * m.r[ac][i, j]) * m.q[t, i, j, c]
        == 0
    )

def regulator_v_drop_rule(m: ConcreteModel, t, i, j, ph):
    return (
        m.v_reg[t, j, ph]
        - m.v[t, j, ph]
        + 2 * (m.r[ph + ph][i, j] * m.p[t, i, j, ph] + m.x[ph + ph][i, j] * m.q[t, i, j, ph])
        == 0
    )

def regulator_tap_rule(m: ConcreteModel, t, i, j, ph):
    return (
        m.v_reg[t, j, ph] == m.v[t, i, ph]*m.ratio[j, ph]**2
    )

def include_regulator_equation(cm: ConcreteModel):
    cm.regulator_v_drop = Constraint(
        cm.t_set, cm.reg_set, cm.phase_set, rule=regulator_v_drop_rule
    )
    cm.regulator_tap = Constraint(
        cm.t_set, cm.reg_set, cm.phase_set, rule=regulator_tap_rule
    )



def p_control_p_gen_rule(m: ConcreteModel, t, j, ph):
    return m.p_gen[t, j, ph] <= m.p_gen_setpoint[t, j, ph]


def p_control_q_gen_rule(m: ConcreteModel, t, j, ph):
    return m.q_gen[t, j, ph] == m.q_gen_setpoint[t, j, ph]


def q_control_p_gen_rule(m: ConcreteModel, t, j, ph):
    return m.p_gen[t, j, ph] == m.p_gen_setpoint[t, j, ph]


def q_control_q_gen_rule(m: ConcreteModel, t, j, ph):
    q_max = sqrt(m.s_gen_rated[j, ph] ** 2 - m.p_gen_setpoint[t, j, ph] ** 2)
    return inequality(-q_max, m.q_gen[t, j, ph], q_max)

def manual_q_gen_limit_rule(m: ConcreteModel, t, j, ph):
    q_max_manual = m.q_max_manual[j, ph]
    q_min_manual = m.q_min_manual[j, ph]
    return inequality(q_min_manual, m.q_gen[t, j, ph], q_max_manual)


def pq_control_rule(m: ConcreteModel, t, j, ph):
    control_variable = m.control_variable[j, ph]
    if control_variable == "pq":
        return m.p_gen[t, j, ph] ** 2 + m.q_gen[t, j, ph] ** 2 <= m.s_gen_rated[j, ph] ** 2 
    return Constraint.Skip

def p_gen_rule(m: ConcreteModel, t, j, ph):
    control_variable = m.control_variable[j, ph]
    if control_variable == "p":
        return p_control_p_gen_rule(m, t, j, ph)
    elif control_variable == "q":
        return q_control_p_gen_rule(m, t, j, ph)
    elif control_variable == "pq":
        return pq_control_rule(m, t, j, ph)
    return Constraint.Skip
    
def q_gen_rule(m: ConcreteModel, t, j, ph):
    control_variable = m.control_variable[j, ph]
    if control_variable == "p":
        return p_control_q_gen_rule(m, t, j, ph)
    elif control_variable == "q":
        return q_control_q_gen_rule(m, t, j, ph)
    elif control_variable == "pq":
        return pq_control_rule(m, t, j, ph)
    return Constraint.Skip

def include_gen_equations(cm: ConcreteModel):
    cm.manual_q_gen_limit = Constraint(
        cm.t_set, cm.gen_set, cm.phase_set, rule=manual_q_gen_limit_rule
    )
    cm.p_gen_constraint = Constraint(
        cm.t_set, cm.gen_set, cm.phase_set, rule=p_gen_rule
    )
    cm.q_gen_constraint = Constraint(
        cm.t_set, cm.gen_set, cm.phase_set, rule=q_gen_rule
    )
    cm.pq_gen_constraint = Constraint(
        cm.t_set, cm.gen_set, cm.phase_set, rule=pq_control_rule
    )



def test():
    system_name = "ieee123_30der_bat"
    # filepath = CASES_DIR / ("csv/" + system_name)
    filepath = Path(
        "/mnt/c/Users/gray570/PycharmProjects/pyomo_MPOPF/rawData/IEEE_123_other/csvs"
    )
    assert filepath.exists()
    # Import CSV files
    bus_data = pd.read_csv(Path(filepath / "bus_data.csv"))
    branch_data = pd.read_csv(Path(filepath / "branch_data.csv"))
    cap_data = pd.read_csv(Path(filepath / "cap_data.csv"))
    # reg_data = pd.read_csv(Path(filepath / "reg_data.csv"))
    gen_data = pd.read_csv(Path(filepath / "gen_data.csv"))
    bat_data = pd.read_csv(Path(filepath / "battery_data.csv"))
    # bat_data = pd.read_csv(Path(filepath / "battery_data_v1.csv"))
    default_loadshape = pd.read_csv(Path(filepath / "default_loadshape.csv"))
    pv_shape = pd.read_csv(Path(filepath / "pv_loadshape.csv"))
    schedules = pd.DataFrame(columns=["time", "PV", "default", "price"])
    schedules.time = default_loadshape["time"].to_numpy() - 1
    schedules.PV = pv_shape["PV"].to_numpy()
    schedules.default = default_loadshape["M"].to_numpy()
    bus_data["load_shape"] = "default"
    # schedules = pd.read_csv(Path(filepath / "schedules.csv"))
    m = PyoMP(
        branch_data=branch_data,
        bus_data=bus_data,
        gen_data=gen_data,
        # reg_data=reg_data,
        cap_data=cap_data,
        bat_data=bat_data,
        schedules=schedules,
        start_step=0,
        n_steps=24,
        delta_t=1,
    )
    m.build()
    m.solve(loss_minimize_with_scd)
    print(m.get_voltages())
    print(m.get_apparent_power_flows())
    print(m.get_p_gens())
    print(m.get_q_gens())
    print(m.get_p_charge())
    print(m.get_p_discharge())
    print(m.get_soc())
    fig = plot_voltages_timeseries(m.get_voltages())
    fig.show()
    fig = plot_gens_timeseries(m.get_p_gens(), m.get_q_gens())
    fig.show()
    fig = plot_device_timeseries(m.get_p_charge())
    fig.show()


if __name__ == "__main__":
    test()
