
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
import math
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Param,
    Set,
    NonNegativeReals,
    minimize,
    sqrt,
    inequality,
    Binary,
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


class BaseModelMP:
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
        self.cm = ConcreteModel()
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

    def import_generator_data(self):
        ## parsing gen_data
        self.cm.p_gen_setpoint = {}
        self.cm.q_gen_setpoint = {}
        self.cm.s_gen_rated = {}
        if self.gen is not None:
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

    def import_battery_data_old(self):
        phases = ["a", "b", "c"]
        ## parsing bat_data
        self.cm.bat_set = []
        self.cm.p_bat_max = {}
        self.cm.s_bat_max = {}
        self.cm.charge_efficiency = {}
        self.cm.discharge_efficiency = {}
        self.cm.charge_state_min = {}
        self.cm.charge_state_max = {}
        self.cm.charge_state_initial = {}
        if self.bat is not None:
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
        self.cm.p_loads = Param(
            self.cm.t_set,
            self.cm.node_set,
            self.cm.phase_set,
            initialize=self.p_load,
            mutable=True,
        )
        self.cm.q_loads = Param(
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
        self.cm.reg_set = sorted(set(zip(self.reg["fb"], self.reg["tb"])))
        self.cm.bat_set = sorted(set(self.bat["id"]))
        self.cm.phase_set = {"a", "b", "c"}
        self.cm.swing_bus_set = {self.swing_bus_id}

        ## initializing model parameters
        self.cm.n_steps = self.n_steps
        self.cm.swing_bus = self.swing_bus_id
        self.cm.r = self.r
        self.cm.x = self.x
        self.cm.cost = self.schedules.price

        self.import_bus_data()
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
        # self.cm.p_gen = Var(self.cm.t_set, self.cm.gen_set, self.cm.phase_set)
        self.cm.q_gen = Var(self.cm.t_set, self.cm.gen_set, self.cm.phase_set)
        self.cm.p_charge = Var(
            self.cm.t_set, self.cm.bat_set, self.cm.phase_set, domain=NonNegativeReals
        )
        self.cm.p_discharge = Var(
            self.cm.t_set, self.cm.bat_set, self.cm.phase_set, domain=NonNegativeReals
        )
        self.cm.charge_state = Var(
            self.cm.t_set, self.cm.bat_set, self.cm.phase_set, domain=NonNegativeReals
        )


def test():
    system_name = "ieee123_30der_bat"
    filepath = CASES_DIR / ("csv/" + system_name)
    # Import CSV files
    bus_data = pd.read_csv(Path(filepath / "bus_data.csv"))
    branch_data = pd.read_csv(Path(filepath / "branch_data.csv"))
    cap_data = pd.read_csv(Path(filepath / "cap_data.csv"))
    reg_data = pd.read_csv(Path(filepath / "reg_data.csv"))
    gen_data = pd.read_csv(Path(filepath / "gen_data.csv"))
    bat_data = pd.read_csv(Path(filepath / "battery_data.csv"))
    schedules = pd.read_csv(Path(filepath / "schedules.csv"))
    m = BaseModelMP(
        branch_data=branch_data,
        bus_data=bus_data,
        gen_data=gen_data,
        reg_data=reg_data,
        cap_data=cap_data,
        bat_data=bat_data,
        schedules=schedules,
        start_step=0,
        n_steps=24,
        delta_t=1,
    )
    m.build()


if __name__ == "__main__":
    test()
