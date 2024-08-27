from distopf.glmpy.basic import Gridlabd
import os
import pandas as pd
import numpy as np
import csv
from pathlib import Path
import distopf as opf


# from pandas.core.frame import DataFrame

class Csv2glm:
    def __init__(
            self,
            output_name,
            branch_data=None,
            bus_data=None,
            gen_data=None,
            cap_data=None,
            reg_data=None,
            p_gen_result=None,
            q_gen_result=None,
            seed_model=None,
            down_nodes=None,
            make_p_out_player=False,
            make_p_max_player=False,
            make_up_player=False,
            model_results_out_dir="",
            cvr=None,
            single_run=False,
            helics_config=None,
            gen_mult=None,
            gen_mult_for_p_max_only=False,
            q_gen_mult=None,
            load_mult=None,
            rating_mult=1.2,
            multiplier_update_period=60,
            opf_model=None,
            tz="PST+8PDT",
            starttime="'2001-08-01 12:00:00'",
            stoptime="'2001-08-01 12:00:00'",
    ):
        self.model_results_out_dir = Path(model_results_out_dir).as_posix()
        # ~~~~~~~~~~~~~~~~~~~~ Load Data Frames ~~~~~~~~~~~~~~~~~~~~
        self.branch_data = opf.handle_branch_input(branch_data)
        self.bus_data = opf.handle_bus_input(bus_data)
        self.gen_data = opf.handle_gen_input(gen_data)
        self.cap_data = opf.handle_cap_input(cap_data)
        self.reg_data = opf.handle_reg_input(reg_data)
        self.p_gen_result = p_gen_result
        self.q_gen_result = q_gen_result
        self.gen_mult_for_p_max_only = gen_mult_for_p_max_only
        if down_nodes is None:
            self.down_nodes = []
        else:
            self.down_nodes = down_nodes
        if cvr is not None:
            self.cvr = np.array(cvr)
        else:
            self.cvr = None
        self.swing_index = bus_data.loc[bus_data.bus_type.str.contains(opf.SWING_BUS)].index[0]
        self.swing_name = f"n{self.bus_data.at[self.swing_index, 'name']}"
        self.v_ln_base = bus_data.at[self.swing_index, "v_ln_base"]
        self.v_up = bus_data.loc[self.swing_index, ["v_a", "v_b", "v_c"]].to_numpy() * self.v_ln_base
        # self.s_dn_pu = s_dn_pu
        self.s_base = bus_data.at[self.swing_index, "s_base"]
        if isinstance(gen_mult, (os.PathLike, str)):
            self.gen_mult = Path(gen_mult)
        else:
            self.gen_mult = gen_mult
        if q_gen_mult is not None:
            if type(q_gen_mult) in [list, tuple, np.ndarray]:
                self.q_gen_mult = [Path(fn) for fn in q_gen_mult]
            else:
                self.q_gen_mult = Path(q_gen_mult)
        else:
            self.q_gen_mult = None

        if isinstance(load_mult, (os.PathLike, str)):
            self.load_mult = Path(load_mult)
        else:
            self.load_mult = load_mult
        self.multiplier_update_period = multiplier_update_period
        self.opf_model = opf_model
        self.rating_mult = rating_mult
        self.tz = tz
        self.starttime = starttime.strip("'").strip('"')
        self.stoptime = stoptime.strip("'").strip('"')

        if seed_model is None:
            self.glm = Gridlabd()
            self.glm.model = {}
            self.glm.clock = {
                "timezone": self.tz,
                "starttime": "'" + self.starttime + "'",
                "stoptime": "'" + self.stoptime + "'",
            }
            self.glm.directives = [
                # '#set suppress_repeat_messages=1',
                "#set relax_naming_rules=1",
                "#set profiler=0",
                "#set double_format=%+.12lg",
                "#set complex_format=%+.12lg%+.12lf%c",
                "#set verbose=0",
            ]
            self.glm.modules = {"tape": {}, "powerflow": {}, "generators": {}}
            self.glm.classes = {}
            self.glm.schedules = {}
        else:
            self.glm = Gridlabd(seed_model, os.getcwd())

        # add helics_msg
        if helics_config is not None:
            self.glm.modules["connection"] = {}
            self.glm.model["helics_msg"] = {
                "GLD1": {"configure": Path(helics_config).as_posix()}
            }

        # Add links
        for i in range(len(self.branch_data)):
            from_bus = int(self.branch_data.loc[i, "fb"])
            to_bus = int(self.branch_data.loc[i, "tb"])
            r = self.branch_data.loc[i, ["raa", "rab", "rac", "rbb", "rbc", "rcc"]] * self.branch_data.loc[i, "z_base"]
            x = self.branch_data.loc[i, ["xaa", "xab", "xac", "xbb", "xbc", "xcc"]] * self.branch_data.loc[i, "z_base"]
            r = r.to_numpy().flatten()
            x = x.to_numpy().flatten()
            link_type = self.branch_data.at[i, "type"]
            status = self.branch_data.at[i, "status"]
            phases = self.branch_data.at[i, "phases"]
            if to_bus in self.reg_data.tb.to_numpy():
                self.make_regulator(from_bus, to_bus)
            elif link_type in ["switch", "recloser", "fuse"]:
                self.make_switch(from_bus, to_bus, phases, status)
            else:
                config_name, phases = self.make_line_config(from_bus, to_bus, r, x)
                self.make_overhead_line(from_bus, to_bus, phases, config_name)
        # Add buses
        for i in range(len(self.bus_data)):
            self.make_node(i)

        # Add additional players, recorders
        v_prop = "voltage_A.real,voltage_A.imag,voltage_B.real,voltage_B.imag,voltage_C.real,voltage_C.imag"
        if make_up_player:
            for ph in ["A", "B", "C"]:
                self.make_player(
                    self.swing_name,
                    f"{self.model_results_out_dir}/v_in_{ph}.player",
                    f"voltage_{ph}",
                )
        # for down_node in self.down_nodes:
        #     down_node = int(down_node)
        #     self.make_recorder(
        #         f"node_{int(down_node)}",
        #         f"{self.model_results_out_dir}/v{down_node}.csv",
        #         v_prop,
        #     )
        # Add p and q recorder
        pq_prop = "measured_power.real,measured_power.imag,"
        pq_a_prop = "measured_power_A.real,measured_power_A.imag,"
        pq_b_prop = "measured_power_B.real,measured_power_B.imag,"
        pq_c_prop = "measured_power_C.real,measured_power_C.imag"
        s_prop = pq_prop + pq_a_prop + pq_b_prop + pq_c_prop
        self.make_recorder(self.swing_name, f"{self.model_results_out_dir}/s_in.csv", s_prop)
        # Add a collector, voltdump, and currdump
        self.glm.model["collector"] = {
            "power_loss_collector": {
                "group": "class=overhead_line",
                "property": "sum(power_losses.real)",
                "interval": "1",
                "file": f"{self.model_results_out_dir}/power_losses.csv",
            },
            "reactive_power_loss_collector": {
                "group": "class=overhead_line",
                "property": "sum(power_losses.imag)",
                "interval": "1",
                "file": f"{self.model_results_out_dir}/var_losses.csv",
            },
            "substation_power_collector": {
                "group": "class=meter and groupid=0",
                "property": "sum(measured_power.real)",
                "interval": "1",
                "file": f"{self.model_results_out_dir}/sub_power.csv",
            },
        }
        self.glm.model["voltdump"] = {
            "voltdump_1": {
                "filename": f"{self.model_results_out_dir}/output_voltage.csv",
                # 'group': '1',
                "mode": "POLAR",
            }
        }
        self.glm.model["currdump"] = {
            "currdump_1": {
                "filename": f"{self.model_results_out_dir}/output_current.csv",
                # 'mode': 'POLAR'
            }
        }
        self.glm.model["group_recorder"] = {
            "grp_rec_vA_mag": {
                "group": "class=meter",
                "property": "voltage_A",
                "complex_part": "MAG",
                "file": f"{self.model_results_out_dir}/grp_rec_vA_mag.csv",
                "interval": "1",
            },
            "grp_rec_vB_mag": {
                "group": "class=meter",
                "property": "voltage_B",
                "complex_part": "MAG",
                "file": f"{self.model_results_out_dir}/grp_rec_vB_mag.csv",
                "interval": "1",
            },
            "grp_rec_vC_mag": {
                "group": "class=meter",
                "property": "voltage_C",
                "complex_part": "MAG",
                "file": f"{self.model_results_out_dir}/grp_rec_vC_mag.csv",
                "interval": "1",
            },
            "grp_rec_vA_angle": {
                "group": "class=meter",
                "property": "voltage_A",
                "complex_part": "ANG_DEG",
                "file": f"{self.model_results_out_dir}/grp_rec_vA_angle.csv",
                "interval": "1",
            },
            "grp_rec_vB_angle": {
                "group": "class=meter",
                "property": "voltage_B",
                "complex_part": "ANG_DEG",
                "file": f"{self.model_results_out_dir}/grp_rec_vB_angle.csv",
                "interval": "1",
            },
            "grp_rec_vC_angle": {
                "group": "class=meter",
                "property": "voltage_C",
                "complex_part": "ANG_DEG",
                "file": f"{self.model_results_out_dir}/grp_rec_vC_angle.csv",
                "interval": "1",
            },
            "grp_rec_pA": {
                "group": "class=overhead_line",
                "property": "power_in_A",
                "complex_part": "REAL",
                "file": f"{self.model_results_out_dir}/grp_rec_pA.csv",
                "interval": "1",
            },
            "grp_rec_pB": {
                "group": "class=overhead_line",
                "property": "power_in_B",
                "complex_part": "REAL",
                "file": f"{self.model_results_out_dir}/grp_rec_pB.csv",
                "interval": "1",
            },
            "grp_rec_pC": {
                "group": "class=overhead_line",
                "property": "power_in_C",
                "complex_part": "REAL",
                "file": f"{self.model_results_out_dir}/grp_rec_pC.csv",
                "interval": "1",
            },
            "grp_rec_qA": {
                "group": "class=overhead_line",
                "property": "power_in_A",
                "complex_part": "IMAG",
                "file": f"{self.model_results_out_dir}/grp_rec_qA.csv",
                "interval": "1",
            },
            "grp_rec_qB": {
                "group": "class=overhead_line",
                "property": "power_in_B",
                "complex_part": "IMAG",
                "file": f"{self.model_results_out_dir}/grp_rec_qB.csv",
                "interval": "1",
            },
            "grp_rec_qC": {
                "group": "class=overhead_line",
                "property": "power_in_C",
                "complex_part": "IMAG",
                "file": f"{self.model_results_out_dir}/grp_rec_qC.csv",
                "interval": "1",
            },
        }
        self.glm.model["impedance_dump"] = {
            "impedance_dump_1": {
                "filename": f"{self.model_results_out_dir}/impedance.xml",
            }
        }
        if not single_run:
            self.glm.model["voltdump"]["voltdump_1"]["runtime"] = "2001-08-01 12:00:01"
        output_name_str = Path(output_name).as_posix()
        self.glm.write(output_name_str)
        self.output_name = output_name_str
        # glmanip.write(
        #     output_name_str, self.glm.model, self.glm.clock, self.glm.directives, self.glm.modules, self.glm.classes, self.glm.schedules)
        # return output_name, self.glm

    def make_node(self, i):
        bus_id = self.bus_data.at[i, "id"]
        target_dict = self.glm.model
        phases = self.bus_data.at[i, "phases"]
        if target_dict.get("load") is None:
            target_dict["load"] = {}
        if target_dict.get("meter") is None:
            target_dict["meter"] = {}

        name = f"n{self.bus_data.at[i, 'name']}"
        cvr_p = self.bus_data.at[i, "cvr_p"]
        cvr_q = self.bus_data.at[i, "cvr_q"]
        loads = self.bus_data.loc[i, ["pl_a", "pl_b", "pl_c", "ql_a", "ql_b", "ql_c"]].to_numpy().flatten()
        load_exists = np.max(np.abs(loads)) > 0
        gen_exists = bus_id in self.gen_data.id.to_numpy()
        cap_exists = bus_id in self.cap_data.id.to_numpy()

        # ###### Create parent node for load, gen, and cap ######
        target_dict["meter"][name] = {
            "phases": f"{phases.upper()}N",
            "nominal_voltage": str(self.v_ln_base),
            "groupid": "1",
        }
        # v_prop = 'measured_voltage_A,measured_voltage_B,measured_voltage_C'
        # make_recorder(name, f'{self.model_results_out_dir}/v{int(node_id)}.csv', v_prop, target_dict)
        if i == self.swing_index:  # assume first bus is swing bus
            target_dict["meter"][name]["bustype"] = "SWING"
            target_dict["meter"][name]["voltage_A"] = f"{self.v_up[0]}+0d"
            target_dict["meter"][name]["voltage_B"] = f"{self.v_up[1]}-120d"
            target_dict["meter"][name]["voltage_C"] = f"{self.v_up[2]}+120d"
            target_dict["meter"][name]["groupid"] = "0"

        if load_exists:
            if cvr_p == 0 and cvr_q == 0:  # Use constant power loads
                self.make_const_load(bus_id, name)
            else:  # Use ZIP loads
                self.make_zip_load(bus_id, name)
        # ###### Add child inverter object to represent generators. #######
        if gen_exists:
            self.make_inverter(bus_id, name)
        # ###### Add child inverter object to represent battery. #######
        # if p_bess is not None:
        #     bess_exists = np.max(np.abs(np.array(p_bess).flatten())) > 0
        #     if bess_exists:
        #         self.make_bess_inverter(bus_id, phases, p_bess, name)
        # ###### Add child capacitor object to represent capacitors. #######
        if cap_exists:
            self.make_cap(bus_id, parent=name)
        return name

    def make_const_load(self, bus_id, parent):
        target_dict = self.glm.model
        i = bus_id - 1
        s_base = self.bus_data.at[i, "s_base"]
        p_l = self.bus_data.loc[i, ["pl_a", "pl_b", "pl_c"]] * s_base
        q_l = self.bus_data.loc[i, ["ql_a", "ql_b", "ql_c"]] * s_base
        p_load = np.array(p_l)
        q_load = np.array(q_l)
        phases = self.bus_data.at[i, "phases"]
        s_str = ["0", "0", "0"]
        if isinstance(self.load_mult, int) or isinstance(self.load_mult, float):
            p_load = p_load * self.load_mult
            q_load = q_load * self.load_mult
        s = np.array(p_load + 1j * q_load).flatten()
        for i in range(3):  # for each phase
            if s[i] != 0:
                s_str[i] = str(s[i]).strip("()")  # remove parenthesis

        if target_dict.get("load") is None:
            target_dict["load"] = {}
        target_dict["load"][f"load_{int(bus_id)}"] = {
            "parent": parent,
            "phases": phases.upper(),
            "nominal_voltage": str(self.v_ln_base),
            "constant_power_A": s_str[0],
            "constant_power_B": s_str[1],
            "constant_power_C": s_str[2],
        }
        if self.load_mult is not None and not (
                isinstance(self.load_mult, int) or isinstance(self.load_mult, float)
        ):
            p = Path(self.load_mult)
            base_dir = p.parent
            for ph in "ABC":
                player_file = base_dir / "players" / f"load_{int(bus_id)}{ph}.player"
                self.make_player_file(
                    self.load_mult, player_file, s, time_delta=self.multiplier_update_period // 60, time_delta_unit="m"
                )
                self.make_player(
                    parent, player_file.relative_to(base_dir), f"constant_power_{ph}"
                )

    def make_zip_load(self, bus_id, parent):
        target_dict = self.glm.model
        i = bus_id - 1
        s_base = self.bus_data.at[i, "s_base"]
        p_l = self.bus_data.loc[i, ["pl_a", "pl_b", "pl_c"]] * s_base
        q_l = self.bus_data.loc[i, ["ql_a", "ql_b", "ql_c"]] * s_base
        p_load = np.array(p_l)
        q_load = np.array(q_l)
        phases = self.bus_data.at[i, "phases"]
        cvr_p = self.bus_data.at[bus_id - 1, "cvr_p"]
        cvr_q = self.bus_data.at[bus_id - 1, "cvr_q"]
        cvr = np.array([cvr_p, cvr_q])
        const_s = 1 - cvr / 2
        const_i = np.array([0, 0])
        const_z = self.cvr / 2
        if np.isscalar(self.load_mult):
            p_load = p_load * self.load_mult
            q_load = q_load * self.load_mult
        if target_dict.get("load") is None:
            target_dict["load"] = {}
        target_dict["load"][f"load_{int(bus_id)}_p"] = {
            "parent": parent,
            "phases": phases.upper(),
            "nominal_voltage": str(self.v_ln_base),
            "base_power_A": str(p_load[0]),
            "base_power_B": str(p_load[1]),
            "base_power_C": str(p_load[2]),
            "power_pf_A": str(1),
            "power_pf_B": str(1),
            "power_pf_C": str(1),
            "impedance_pf_A": str(1),
            "impedance_pf_B": str(1),
            "impedance_pf_C": str(1),
            "power_fraction_A": str(const_s[0]),
            "power_fraction_B": str(const_s[0]),
            "power_fraction_C": str(const_s[0]),
            "current_fraction_A": str(const_i[0]),
            "current_fraction_B": str(const_i[0]),
            "current_fraction_C": str(const_i[0]),
            "impedance_fraction_A": str(const_z[0]),
            "impedance_fraction_B": str(const_z[0]),
            "impedance_fraction_C": str(const_z[0]),
        }
        target_dict["load"][f"load_{int(bus_id)}_q"] = {
            "parent": parent,
            "phases": phases.upper(),
            "nominal_voltage": str(self.v_ln_base),
            "base_power_A": str(q_load[0]),
            "base_power_B": str(q_load[1]),
            "base_power_C": str(q_load[2]),
            "power_pf_A": str(0),
            "power_pf_B": str(0),
            "power_pf_C": str(0),
            "impedance_pf_A": str(0),
            "impedance_pf_B": str(0),
            "impedance_pf_C": str(0),
            "power_fraction_A": str(const_s[1]),
            "power_fraction_B": str(const_s[1]),
            "power_fraction_C": str(const_s[1]),
            "current_fraction_A": str(const_i[1]),
            "current_fraction_B": str(const_i[1]),
            "current_fraction_C": str(const_i[1]),
            "impedance_fraction_A": str(const_z[1]),
            "impedance_fraction_B": str(const_z[1]),
            "impedance_fraction_C": str(const_z[1]),
        }

        if self.load_mult is not None and not np.isscalar(self.load_mult):
            p = Path(self.load_mult)
            base_dir = p.parent
            for index, ph in enumerate("ABC"):
                # P Loads
                player_file = (
                        base_dir / "players" / f"load_{int(bus_id)}_p{ph}.player"
                ).as_posix()
                self.make_player_file(
                    self.load_mult,
                    player_file,
                    p_load[index],
                    time_delta=self.multiplier_update_period // 60,
                    time_delta_unit="m",
                )
                self.make_player(
                    f"load_{int(bus_id)}_p",
                    Path(player_file).relative_to(base_dir),
                    f"base_power_{ph}",
                )
                # Q Loads
                player_file = (
                        base_dir / "players" / f"load_{int(bus_id)}_q{ph}.player"
                ).as_posix()
                self.make_player_file(
                    self.load_mult,
                    player_file,
                    q_load[index],
                    time_delta=self.multiplier_update_period // 60,
                    time_delta_unit="m",
                )
                self.make_player(
                    f"load_{int(bus_id)}_q",
                    Path(player_file).relative_to(base_dir),
                    f"base_power_{ph}",
                )

    def make_inverter(self, bus_id, parent):

        i = bus_id - 1
        s_base = self.bus_data.at[i, "s_base"]
        p_max = self.gen_data.loc[i, ["pa", "pb", "pc"]] * s_base
        if self.p_gen_result is not None:
            p_gen = self.p_gen_result.loc[i, ["a", "b", "c"]] * s_base
        else:
            p_gen = p_max
        if self.q_gen_result is not None:
            q_gen = self.q_gen_result.loc[i, ["a", "b", "c"]] * s_base
        else:
            q_gen = self.gen_data.loc[i, ["qa", "qb", "qc"]] * s_base
        s_rated = self.gen_data.loc[i, ["sa_max", "sb_max", "sc_max"]] * s_base
        phases = self.bus_data.at[i, "phases"]

        target_dict = self.glm.model
        if target_dict.get("inverter") is None:
            target_dict["inverter"] = {}
        if target_dict.get("meter") is None:
            target_dict["meter"] = {}
        target_dict["meter"][f"inv_{int(bus_id)}_meter"] = {
            "parent": parent,
            "phases": f"{phases.upper()}N",
            "nominal_voltage": str(self.v_ln_base),
        }

        self.make_recorder(
            f"inv_{int(bus_id)}_meter",
            f"{self.model_results_out_dir}/inv_{int(bus_id)}_meter.csv",
            "measured_power_A,measured_power_B,measured_power_C",
        )

        if p_gen is None:
            p_gen = p_max
            if np.isscalar(self.gen_mult):  # no player is used so apply gen_mult now
                p_gen = np.array(p_max) * self.gen_mult
        if q_gen is None:
            q_gen = np.array([0, 0, 0])
        assert len(q_gen) == 3, "q_gen should be a list of 3 values (3 phases)."
        if np.isscalar(self.q_gen_mult):
            q_gen = np.array(q_gen) * self.q_gen_mult
        for i, ph in enumerate("ABC"):
            if ph not in phases.upper():
                continue
            name = f"inv_{int(bus_id)}{ph}"
            target_dict["inverter"][name] = {
                "parent": f"inv_{int(bus_id)}_meter",
                "phases": f"{ph}N".upper(),
                "rated_power": str(list(s_rated)[i]),
                "inverter_type": "FOUR_QUADRANT",
                "four_quadrant_control_mode": "CONSTANT_PQ",
                "generator_status": "ONLINE",
                "generator_mode": "CONSTANT_PF",
                "P_Out": str(list(p_gen)[i]),
                "Q_Out": str(list(q_gen)[i]),
                "V_In": "1000",  # V_In*I_in*inverter_efficiency defines maximum power output.
                "I_In": "1000",
                "inverter_efficiency": "0.99",
            }
            if self.gen_mult is not None and not np.isscalar(self.gen_mult):
                p = Path(self.gen_mult)
                base_dir = p.parent
                if self.gen_mult_for_p_max_only:
                    # if self.opf_model == "curtail":
                    # Create player for and I_in
                    player_file = base_dir / "players" / f"{name}_i_in.player"
                    self.make_player_file(
                        self.gen_mult,
                        player_file,
                        list(p_max)[i] / (1000 * 0.99),
                        time_delta=self.multiplier_update_period // 60,
                        time_delta_unit="m",
                    )
                    self.make_player(
                        name, player_file.relative_to(base_dir), "I_In"
                    )
                return
                # elif self.opf_model != "p_target":
                # Create player for P_Out
                player_file = base_dir / "players" / f"{name}.player"
                self.make_player_file(
                    self.gen_mult,
                    player_file,
                    list(p_gen)[i],
                    time_delta=self.multiplier_update_period // 60,
                    time_delta_unit="m",
                )
                self.make_player(
                    name, player_file.relative_to(base_dir), "P_Out"
                )

            if self.q_gen_mult is not None and not np.isscalar(self.q_gen_mult):
                if type(self.q_gen_mult) in [list, tuple, np.ndarray]:
                    p = Path(self.q_gen_mult[i])
                    q_file_name = self.q_gen_mult[i]
                else:
                    p = Path(self.q_gen_mult)
                    q_file_name = self.q_gen_mult
                base_dir = p.parent
                player_file = base_dir / "players" / f"{name}q.player"
                q_lim = np.sqrt(
                    (list(p_max)[i] * self.rating_mult) ** 2
                    - list(p_max)[i] ** 2
                )
                self.make_player_file(
                    q_file_name,
                    player_file,
                    list(q_gen)[i],
                    time_delta=self.multiplier_update_period // 60,
                    time_delta_unit="m",
                )
                self.make_player(name, player_file.relative_to(base_dir), "Q_Out")

    # def make_bess_inverter(
    #         self, node_id, phases, p_rated, parent, p_gen=None, q_gen=None
    # ):
    #     if q_gen is None:
    #         q_gen = np.array([0, 0, 0])
    #     if p_gen is None:
    #         p_gen = np.array([0, 0, 0])
    #     assert len(q_gen) == 3, "q_gen should be a list of 3 values (3 phases)."
    #     target_dict = self.glm.model
    #     if target_dict.get("inverter") is None:
    #         target_dict["inverter"] = {}
    #     if target_dict.get("meter") is None:
    #         target_dict["meter"] = {}
    #     if np.isscalar(self.gen_mult) and p_gen is None:
    #         p_gen = np.array(p_rated) * self.gen_mult
    #     if np.isscalar(self.q_gen_mult):
    #         q_gen = np.array(q_gen) * self.q_gen_mult
    #     target_dict["meter"][f"bess_inv_{int(node_id)}_meter"] = {
    #         "parent": parent,
    #         "phases": phases,
    #         "nominal_voltage": str(self.v_ln_base),
    #     }
    #     # make_recorder(f'inv_{int(node_id)}_meter', f'{self.model_results_out_dir}/inv_{int(node_id)}_meter.csv',
    #     #               'measured_power_A,measured_power_B,measured_power_C', target_dict)
    #     self.make_recorder(
    #         f"bess_inv_{int(node_id)}_meter",
    #         f"{self.model_results_out_dir}/bess_inv_{int(node_id)}_meter.csv",
    #         "measured_power_A,measured_power_B,measured_power_C",
    #     )
    #     for i, ph in enumerate("ABC"):
    #         if ph in phases:
    #             name = f"bess_inv_{int(node_id)}{ph}"
    #             target_dict["inverter"][name] = {
    #                 "parent": f"bess_inv_{int(node_id)}_meter",
    #                 "phases": f"{ph}N",
    #                 "rated_power": str(
    #                     list(p_rated)[i] * self.rating_mult
    #                 ),  # 1.2*P_out
    #                 "inverter_type": "FOUR_QUADRANT",
    #                 "four_quadrant_control_mode": "CONSTANT_PQ",
    #                 "generator_status": "ONLINE",
    #                 "generator_mode": "CONSTANT_PF",
    #                 "P_Out": str(list(p_gen)[i]),
    #                 "Q_Out": str(list(q_gen)[i]),
    #                 "V_In": "1000",  # V_In*I_in*inverter_efficiency defines maximum power output.
    #                 "I_In": "1000",
    #                 "inverter_efficiency": "0.99",
    #             }
    #             if self.gen_mult is not None and not np.isscalar(self.gen_mult):
    #                 p = Path(self.gen_mult)
    #                 base_dir = p.parent
    #
    #                 if self.opf_model == "curtail":
    #                     # Create player for and I_in
    #                     player_file = base_dir / "players" / f"{name}_i_in.player"
    #                     self.make_player_file(
    #                         self.gen_mult,
    #                         player_file,
    #                         list(p_rated)[i] / (1000 * 0.99),
    #                         time_delta=self.multiplier_update_period // 60,
    #                         time_delta_unit="m",
    #                     )
    #                     self.make_player(
    #                         name, player_file.relative_to(base_dir), "I_In"
    #                     )
    #                 elif self.opf_model != "p_target":
    #                     # Create player for P_Out
    #                     player_file = base_dir / "players" / f"{name}.player"
    #                     self.make_player_file(
    #                         self.gen_mult,
    #                         player_file,
    #                         list(p_rated)[i],
    #                         time_delta=self.multiplier_update_period // 60,
    #                         time_delta_unit="m",
    #                     )
    #                     self.make_player(
    #                         name, player_file.relative_to(base_dir), "P_Out"
    #                     )
    #                     # Create player for rated_power
    #                     player_file = base_dir / "players" / f"{name}_rated.player"
    #                     self.make_player_file(
    #                         self.gen_mult,
    #                         player_file,
    #                         list(p_rated)[i] * self.rating_mult,
    #                         time_delta=self.multiplier_update_period // 60,
    #                         time_delta_unit="m",
    #                     )
    #                     self.make_player(
    #                         name, player_file.relative_to(base_dir), "rated_power"
    #                     )
    #
    #             if self.q_gen_mult is not None and not np.isscalar(self.q_gen_mult):
    #                 if type(self.q_gen_mult) in [list, tuple, np.ndarray]:
    #                     p = Path(self.q_gen_mult[i])
    #                     q_file_name = self.q_gen_mult[i]
    #                 else:
    #                     p = Path(self.q_gen_mult)
    #                     q_file_name = self.q_gen_mult
    #                 base_dir = p.parent
    #                 player_file = base_dir / "players" / f"{name}q.player"
    #                 q_lim = np.sqrt(
    #                     (list(p_rated)[i] * self.rating_mult) ** 2
    #                     - list(p_rated)[i] ** 2
    #                 )
    #                 self.make_player_file(
    #                     q_file_name,
    #                     player_file,
    #                     q_lim,
    #                     time_delta=self.multiplier_update_period // 60,
    #                     time_delta_unit="m",
    #                 )
    #                 self.make_player(name, player_file.relative_to(base_dir), "Q_Out")

    def make_cap(self, bus_id, parent=None):
        target_dict = self.glm.model
        i = bus_id - 1
        s_base = self.bus_data.at[i, "s_base"]
        q_cap = self.cap_data.loc[i, ["qa", "qb", "qc"]] * s_base
        phases = self.bus_data.at[i, "phases"]
        if target_dict.get("capacitor") is None:
            target_dict["capacitor"] = {}

        name = f"cap_{int(bus_id)}"
        target_dict["capacitor"][name] = {
            "phases": phases.upper(),
            "phases_connected": phases.upper(),
            "nominal_voltage": str(self.v_ln_base),
            "capacitor_A": str(list(q_cap)[0]),
            "capacitor_B": str(list(q_cap)[1]),
            "capacitor_C": str(list(q_cap)[2]),
            "control": "MANUAL",
            "control_level": "INDIVIDUAL",
            "switchA": "CLOSED",
            "switchB": "CLOSED",
            "switchC": "CLOSED",
        }
        if parent is not None:
            target_dict["capacitor"][name]["parent"] = parent

    def make_line_config(self, f_bus, to_bus, r_updiag, x_updiag):
        target_dict = self.glm.model
        line_config_name = f"line_config_{int(f_bus)}_{int(to_bus)}"
        if target_dict.get("line_configuration") is None:
            target_dict["line_configuration"] = {}
        z_bus = ["0"] * 6  # list of 9 zeros
        for index in range(6):
            z = r_updiag[index] + 1j * x_updiag[index]
            if z != 0:  # remove parenthesis
                z_bus[index] = str(z).strip("(").strip(")")
        target_dict["line_configuration"][str(line_config_name)] = {
            "z11": z_bus[0],
            "z12": z_bus[1],
            "z13": z_bus[2],
            "z21": z_bus[1],
            "z22": z_bus[3],
            "z23": z_bus[4],
            "z31": z_bus[2],
            "z32": z_bus[4],
            "z33": z_bus[5],
        }
        phases = ""
        if z_bus[0] != "0":
            phases = phases + "A"
        if z_bus[3] != "0":
            phases = phases + "B"
        if z_bus[5] != "0":
            phases = phases + "C"
        phases = phases + "N"
        return line_config_name, phases.upper()

    def get_phases(self, r_updiag, x_updiag):
        target_dict = self.glm.model
        z_bus = ["0"] * 6  # list of 9 zeros
        for index in range(6):
            z = r_updiag[index] + 1j * x_updiag[index]
            if z != 0:  # remove parenthesis
                z_bus[index] = str(z)[1:-1]
        phases = ""
        if z_bus[0] != "0":
            phases = phases + "A"
        if z_bus[3] != "0":
            phases = phases + "B"
        if z_bus[5] != "0":
            phases = phases + "C"
        phases = phases + "N"
        return phases

    def make_overhead_line(self, from_bus, to_bus, phases, config_name):
        target_dict = self.glm.model
        from_name = f"n{self.bus_data.loc[self.bus_data.id == from_bus, 'name'].to_numpy()[0]}"
        to_name = f"n{self.bus_data.loc[self.bus_data.id == to_bus, 'name'].to_numpy()[0]}"
        line_name = f"oh_line_{int(from_bus)}_{int(to_bus)}"
        if target_dict.get("overhead_line") is None:
            target_dict["overhead_line"] = {}
        target_dict["overhead_line"][str(line_name)] = {
            "phases": phases.upper(),
            "from": from_name,
            "to": to_name,
            "configuration": config_name,
            "length": "5280",  # 1 mile. Config z matrix specified in Ohms but interpreted as ohms/mile.
        }
        return line_name

    def make_switch(self, from_bus, to_bus, phases, status):
        """
        object switch {
            name switch_name;
            from 123;
            to 124;
            phases ABC;
            status OPEN;
        }
        """
        from_name = f"n{self.bus_data.loc[self.bus_data.id == from_bus, 'name'].to_numpy()[0]}"
        to_name = f"n{self.bus_data.loc[self.bus_data.id == to_bus, 'name'].to_numpy()[0]}"
        target_dict = self.glm.model
        link_name = f"switch_{int(from_bus)}_{int(to_bus)}"
        if target_dict.get("switch") is None:
            target_dict["switch"] = {}
        target_dict["switch"][str(link_name)] = {
            "from": from_name,
            "to": to_name,
            "phases": phases.upper(),
            "status": status,
        }
        return link_name

    def make_recloser(self):
        pass

    def make_fuse(self):
        pass

    def make_regulator(self, from_bus, to_bus):
        """
        object regulator {
            name Reg799781;
            phases "ABC";
            from node_799;
            to node_781;
            configuration reg_conf_79978101;
            }
        Parameters
        ----------
        from_bus
        to_bus

        Returns
        -------

        """
        target_dict = self.glm.model
        phases = self.reg_data.loc[self.reg_data.tb == to_bus, "phases"].to_numpy()[0]
        tap_a = self.reg_data.loc[self.reg_data.tb == to_bus, "tap_a"].to_numpy()[0]
        tap_b = self.reg_data.loc[self.reg_data.tb == to_bus, "tap_b"].to_numpy()[0]
        tap_c = self.reg_data.loc[self.reg_data.tb == to_bus, "tap_c"].to_numpy()[0]
        from_name = f"n{self.bus_data.loc[self.bus_data.id == from_bus, 'name'].to_numpy()[0]}"
        to_name = f"n{self.bus_data.loc[self.bus_data.id == to_bus, 'name'].to_numpy()[0]}"
        link_name = f"regulator_{int(from_bus)}_{int(to_bus)}"
        config_name = f"regulator_config_{int(from_bus)}_{int(to_bus)}"
        if target_dict.get("regulator_configuration") is None:
            target_dict["regulator_configuration"] = {}

        target_dict["regulator_configuration"][str(config_name)] = {
            "raise_taps": 16,
            "lower_taps": 16,
            "connect_type": 1,
            "tap_pos_A": tap_a,
            "tap_pos_B": tap_b,
            "tap_pos_C": tap_c,
            "control_level": "INDIVIDUAL",
            "Control": "MANUAL",
        }
        if target_dict.get("regulator") is None:
            target_dict["regulator"] = {}
        target_dict["regulator"][str(link_name)] = {
            "from": from_name,
            "to": to_name,
            "phases": phases.upper(),
            "configuration": config_name,
        }
        pass

    def make_player_file(
            self,
            in_file,
            out_file,
            base_value,
            time_delta=1,
            time_delta_unit="m",
            starttime=None,
    ):
        if starttime is None:
            starttime = self.starttime
        mult = np.genfromtxt(in_file, delimiter=",")
        with open(out_file, "w") as f:
            w = csv.writer(
                f, delimiter=",", quoting=csv.QUOTE_NONE, lineterminator="\n"
            )
            w.writerow([starttime, base_value * mult[0]])
            dt = f"+{int(time_delta)}{time_delta_unit}"
            for i in range(1, len(mult)):
                w.writerow([dt, base_value * mult[i]])

    def make_player(self, parent, file_name, prop):
        target_dict = self.glm.model
        if target_dict.get("player") is None:
            target_dict["player"] = {}
        number = 1
        name = f"{parent}_player_{number}"
        while target_dict["player"].get(name) is not None:
            number += 1
            name = f"{parent}_player_{number}"
        target_dict["player"][name] = {
            "parent": parent,
            "file": Path(file_name).as_posix(),
            "property": prop,
        }

    def make_recorder(self, parent, out_file, prop, interval=1):
        """
        object recorder {
            parent load_38;
            file v38a.csv;
            property voltage_A;
            interval 1;
        };
        """
        target_dict = self.glm.model
        if target_dict.get("recorder") is None:
            target_dict["recorder"] = {}
        number = 1
        name = f"{parent}_recorder_{number}"
        while target_dict["recorder"].get(name) is not None:
            number += 1
            name = f"{parent}_recorder_{number}"
        target_dict["recorder"][name] = {
            "parent": parent,
            "file": out_file,
            "property": prop,
            "interval": str(int(interval)),
        }
