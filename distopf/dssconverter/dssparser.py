from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd


class DSSParser:
    def __init__(
        self,
        dssfile: (str, Path),
        s_base: float = 1e6,
        v_min: float = 0.95,
        v_max: float = 1.05,
        cvr_p: float = 0,
        cvr_q: float = 0,
    ) -> None:
        self.dss = dss
        self.dssfile = dssfile
        self.dss.Text.Command(f"Redirect {self.dssfile}")
        self.s_base = s_base
        self.v_min = v_min
        self.v_max = v_max
        self.cvr_p = cvr_p
        self.cvr_q = cvr_q
        # get dataframes and results
        self.branch_data = self.get_branch_data()
        self.bus_data = self.get_bus_data()
        self.gen_data = self.get_gen_data()
        self.cap_data = self.get_cap_data()
        self.reg_data = self.get_reg_data()
        self.v_solved = self.get_v_solved()
        self.s_solved = self.get_apparent_power_flows()

    def update(self) -> None:
        self.dss.Solution.Solve()
        # get dataframes and results
        self.branch_data = self.get_branch_data()
        self.bus_data = self.get_bus_data()
        self.gen_data = self.get_gen_data()
        self.cap_data = self.get_cap_data()
        self.reg_data = self.get_reg_data()
        self.v_solved = self.get_v_solved()
        self.s_solved = self.get_apparent_power_flows()

    @property
    @cache
    def bus_names(self) -> list[str]:
        """Access all the bus (node) names from the circuit

        Returns:
            list[str]: list of all the bus names
        """
        return self.dss.Circuit.AllBusNames()

    @property
    @cache
    def bus_names_to_index_map(self) -> dict[str, int]:
        """each of the bus mapped to its corresponding index in the bus names list

        Returns:
            dict[str,int]: dictionary with key as bus names and value as its index
        """
        return {bus: index + 1 for index, bus in enumerate(self.bus_names)}

    def bus_names_to_index_map_fun(self, bus: str) -> int:
        return self.bus_names_to_index_map[bus]

    @property
    def basekV_LL(self) -> float:
        """Returns basekV (line to line) of the circuit based on the sourcebus

        Returns:
            float: base kV of the circuit as referred to the source bus
        """
        # make the source bus active before accessing the base kV since there is no provision to get base kV of circuit
        self.dss.Circuit.SetActiveBus(self.source)
        return round(self.dss.Bus.kVBase() * np.sqrt(3), 2)

    @property
    def source(self) -> str:
        """source bus of the circuit.

        Returns:
            str: returns the source bus of the circuit
        """
        # typically the first bus is the source bus
        return self.bus_names[0]

    @property
    # @cache
    def gen_buses(self) -> set[str]:
        flag = self.dss.Generators.First()
        gen_buses = set()
        while flag:
            gen_buses.add(self.dss.Generators.Bus1().split(".")[0])
            flag = self.dss.Generators.Next()
        return gen_buses

    @property
    # @cache
    def cap_buses(self) -> set[str]:
        flag = self.dss.Capacitors.First()
        cap_buses = set()
        while flag:
            cap_buses.add(self.dss.CktElement.BusNames()[0].split(".")[0])
            flag = self.dss.Capacitors.Next()
        return cap_buses

    @property
    # @cache
    def load_buses(self) -> set[str]:
        flag = self.dss.Loads.First()
        load_buses = set()
        while flag:
            load_buses.add(self.dss.CktElement.BusNames()[0].split(".")[0])
            flag = self.dss.Loads.Next()
        return load_buses

    @property
    def num_phase_map(self) -> dict[str, str]:

        # opendss provides nodes phase in number format so we convert it to letter format
        num_phase_mapper = {
            "[1]": "a",
            "[2]": "b",
            "[3]": "c",
            "[1, 2]": "ab",
            "[1, 3]": "ac",
            "[2, 3]": "bc",
            "[1, 2, 3]": "abc",
            "[1, 2, 3, 4]": "abc",  # excluding 4th node
        }
        return num_phase_mapper

    def get_v_solved(self) -> pd.DataFrame:
        va = pd.DataFrame(
            {
                "name": [
                    name.split(".")[0]
                    for name in self.dss.Circuit.AllNodeNamesByPhase(1)
                ],
                "a": self.dss.Circuit.AllNodeVmagPUByPhase(1),
            }
        )
        vb = pd.DataFrame(
            {
                "name": [
                    name.split(".")[0]
                    for name in self.dss.Circuit.AllNodeNamesByPhase(2)
                ],
                "b": self.dss.Circuit.AllNodeVmagPUByPhase(2),
            }
        )
        vc = pd.DataFrame(
            {
                "name": [
                    name.split(".")[0]
                    for name in self.dss.Circuit.AllNodeNamesByPhase(3)
                ],
                "c": self.dss.Circuit.AllNodeVmagPUByPhase(3),
            }
        )
        v_df = pd.merge(va, vb, on="name", how="outer")
        v_df = pd.merge(v_df, vc, on="name", how="outer")
        v_df.index = v_df.name.apply(self.bus_names_to_index_map_fun)
        v_df = v_df.sort_index()
        return v_df

    def get_apparent_power_flows(self) -> pd.DataFrame:
        s_base = self.s_base
        flag = self.dss.PDElements.First()
        power_data = []
        while flag:
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            is_open = [
                self.dss.CktElement.IsOpen(0, ph)
                for ph in range(self.dss.CktElement.NumPhases())
            ]
            if all(is_open):
                flag = self.dss.PDElements.Next()
                continue
            s_out = self._get_powers() * 1000 / self.s_base
            if element_type not in ["line", "transformer"]:
                flag = self.dss.PDElements.Next()
                continue
            bus1 = self.dss.CktElement.BusNames()[0].split(".")[0]
            bus2 = self.dss.CktElement.BusNames()[1].split(".")[0]
            self.dss.Circuit.SetActiveBus(bus2)

            each_power = dict(
                fb=bus1,
                tb=bus2,
                a=s_out[0],
                b=s_out[1],
                c=s_out[2],
            )

            power_data.append(each_power)
            flag = self.dss.PDElements.Next()

        # combine lines between identical buses.
        power_df = pd.DataFrame(power_data)
        power_df = (
            power_df.groupby(by=["fb", "tb"], as_index=False)
            .agg({"fb": "first", "tb": "first", "a": "sum", "b": "sum", "c": "sum"})
            .reset_index(drop=True)
            .sort_values(by=["fb"], ignore_index=True)
            .sort_values(by=["tb"], ignore_index=True)
        )

        return power_df

    def _get_line_zmatrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the z_matrix of a specified line element.

        Returns:
            real z_matrix, imag z_matrix (np.ndarray, np.ndarray): 3x3 numpy array of the z_matrix corresponding to the each of the phases(real,imag)
        """
        n_phases = self.dss.Lines.Phases()

        z_matrix_real = np.zeros((3, 3))
        z_matrix_imag = np.zeros((3, 3))
        if n_phases > 3:
            pass
        if (len(self.dss.CktElement.BusNames()[0].split(".")) == 4) or (
            len(self.dss.CktElement.BusNames()[0].split(".")) == 1
        ):

            # this is the condition check for three phase since three phase is either represented by bus_name.1.2.3 or bus_name
            z_matrix = (
                np.array(self.dss.Lines.RMatrix())
                + 1j * np.array(self.dss.Lines.XMatrix())
            ) * self.dss.Lines.Length()

            z_matrix = z_matrix.reshape(3, 3)

            return np.real(z_matrix), np.imag(z_matrix)

        else:

            # for other than 3 phases
            active_phases = [
                int(phase) for phase in self.dss.CktElement.BusNames()[0].split(".")[1:]
            ]
            z_matrix = np.zeros((3, 3), dtype=complex)
            r_matrix = self.dss.Lines.RMatrix()
            x_matrix = self.dss.Lines.XMatrix()
            counter = 0
            for _, row in enumerate(active_phases):
                for _, col in enumerate(active_phases):
                    z_matrix[row - 1, col - 1] = (
                        complex(r_matrix[counter], x_matrix[counter])
                        * self.dss.Lines.Length()
                    )
                    counter = counter + 1

            return np.real(z_matrix), np.imag(z_matrix)

    def _get_reactor_zmatrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the z_matrix of a specified reactor element.

        Returns:
            real z_matrix, imag z_matrix (np.ndarray, np.ndarray): 3x3 numpy array of the z_matrix corresponding to the each of the phases(real,imag)
        """
        n_phases = self.dss.Reactors.Phases()
        if n_phases == 3:
            return np.eye(3)*self.dss.Reactors.R(), np.eye(3)*self.dss.Reactors.X()

        else:
            # for other than 3 phases
            raise NotImplemented("Parsing reactors with phases other than 3 not implemented")
            # active_phases = [
            #     int(phase) for phase in self.dss.CktElement.BusNames()[0].split(".")[1:]
            # ]
            # z_matrix = np.zeros((3, 3), dtype=complex)
            # r_matrix = self.dss.Reactors.R()
            # x_matrix = self.dss.Reactors.X()
            # counter = 0
            # for _, row in enumerate(active_phases):
            #     for _, col in enumerate(active_phases):
            #         z_matrix[row - 1, col - 1] = (
            #             complex(r_matrix[counter], x_matrix[counter])
            #             * self.dss.Lines.Length()
            #         )
            #         counter = counter + 1

            return np.real(z_matrix), np.imag(z_matrix)

    def _get_powers(self):
        n_phases = self.dss.CktElement.NumPhases()
        pq = np.array(self.dss.CktElement.Powers())
        n_terminals = self.dss.CktElement.NumTerminals()
        n_pq_phases = len(pq) // n_terminals // 2
        pq = pq.reshape(int(n_pq_phases * n_terminals), 2)
        s_out = np.zeros(3, dtype=complex)
        active_phases = np.array([0, 1, 2])
        if n_phases < 3:
            active_phases = (
                np.array(self.dss.CktElement.BusNames()[0].split(".")[1:]).astype(int)
                - 1
            )

        p = pq[:, 0]
        q = pq[:, 1]
        s = p + 1j * q
        s_out_ = -s[n_pq_phases:]
        s_out[active_phases] = s_out_[:n_phases]
        return s_out

    def get_branch_data(self) -> pd.DataFrame:
        s_base = self.s_base
        flag = self.dss.PDElements.First()
        line_data = []
        power_data = []
        while flag:
            switch_status = None
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            element_name = self.dss.CktElement.Name().lower().split(".")[1]
            s_out = self._get_powers()
            z_matrix_real = np.zeros((3, 3))
            z_matrix_imag = np.zeros((3, 3))
            if element_type not in ["line", "transformer", "reactor"]:
                flag = self.dss.PDElements.Next()
                continue
            if element_type == "transformer":
                is_delta = self.dss.Transformers.IsDelta()
                n_windings = self.dss.Transformers.NumWindings()
                r_xfmr = 0
                x_xfmr = 0
                n_phases = self.dss.CktElement.NumPhases()
                n_terminals = self.dss.CktElement.NumTerminals()
                y_prime_flat = np.array(self.dss.CktElement.YPrim())
                y_prim = y_prime_flat[::2] + 1j * y_prime_flat[1::2]
                y_shape = int(np.sqrt(len(y_prim)))
                y_prim = np.reshape(y_prim, (y_shape, y_shape))
                n_y11 = int(y_shape / 2)
                v_all = np.array(self.dss.CktElement.Voltages())
                v_all = v_all[::2] + 1j * v_all[1::2]
                v1 = v_all[: len(v_all) // 2]
                v2 = v_all[len(v_all) // 2 :]
                i_all = np.array(self.dss.CktElement.Currents())
                i_all = i_all[::2] + 1j * i_all[1::2]
                i1 = i_all[: len(i_all) // 2]
                i2 = i_all[len(i_all) // 2 :]
                self.dss.Transformers.Wdg(1)
                # v1 = np.array(self.dss.Transformers.WdgVoltages())
                # v1 = v1[::2] + 1j * v1[1::2]
                kv_h = self.dss.Transformers.kV()
                self.dss.Transformers.Wdg(2)
                # v2 = np.array(self.dss.Transformers.WdgVoltages())
                # v2 = v2[::2] + 1j * v2[1::2]
                kv_l = self.dss.Transformers.kV()
                n = kv_h / kv_l
                y11 = y_prim[:n_y11, :n_y11] * n**2
                y12 = y_prim[:n_y11, n_y11:] * n
                y21 = y_prim[n_y11:, :n_y11]
                y22 = y_prim[n_y11:, n_y11:]
                y_prim_l = np.r_[np.c_[y11, y12], np.c_[y21, y22]]
                # z_prim_l = np.linalg.inv(y_prim_l)
                # z11 = z_prim_l[:n_y11, :n_y11]
                # z12 = z_prim_l[:n_y11, n_y11:]
                # z21 = z_prim_l[n_y11:, :n_y11]
                # z22 = z_prim_l[n_y11:, n_y11:]
                i_all = np.array(self.dss.Transformers.WdgCurrents())
                i_in = i_all[::2]
                i_all = i_all[::2] + i_all[1::2] * 1j
                i_in = i_all[::2]
                i_out = i_all[1::2]
                # i1_in = i_in[::2]
                # i2_in = i_in[1::2]
                # i2_out = i_out[1::2]
                # zabc = (v1[:n_phases] / n - v2[:n_phases]) / -i2[:n_phases]
                # TODO: tranformer model may be wrong but the 13bus results look better with it.
                # if is_delta:
                #     raise Warning("Delta transformer not implemented")
                # if n_windings != 2:
                #
                # for i_wdg in range(1, n_windings + 1):
                # self.dss.Transformers.Wdg(i_wdg)
                v_base_xfmr = self.dss.Transformers.kV() / np.sqrt(3) * 1000
                s_base_xfmr = self.dss.Transformers.kVA() * 1000 / 3
                z_base_xfmr = v_base_xfmr**2 / s_base_xfmr

                x_xfmr = self.dss.Transformers.Xhl() / 100 * z_base_xfmr
                r_xfmr = self.dss.Transformers.R() / 100 * z_base_xfmr * 2
                z_matrix_real[0, 0] = r_xfmr
                z_matrix_real[1, 1] = r_xfmr
                z_matrix_real[2, 2] = r_xfmr
                z_matrix_imag[0, 0] = x_xfmr
                z_matrix_imag[1, 1] = x_xfmr
                z_matrix_imag[2, 2] = x_xfmr
                pass
            if element_type == "line":
                element_name = self.dss.Lines.Name()
                z_matrix_real, z_matrix_imag = self._get_line_zmatrix()

                if self.dss.Lines.IsSwitch():
                    element_type = "switch"
                    switch_status = (
                        "OPEN"
                        if (
                            self.dss.CktElement.IsOpen(1, 0)
                            or self.dss.CktElement.IsOpen(2, 0)
                        )
                        else "CLOSED"
                    )
            if element_type == "reactor":
                element_name = self.dss.Reactors.Name()
                z_matrix_real, z_matrix_imag = self._get_reactor_zmatrix()
            bus1 = self.dss.CktElement.BusNames()[0].split(".")[0]
            bus2 = self.dss.CktElement.BusNames()[1].split(".")[0]
            self.dss.Circuit.SetActiveBus(bus2)
            base_kv_ln = self.dss.Bus.kVBase()
            z_base = (base_kv_ln * 1000) ** 2 / s_base
            line_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]
            line_phases = sorted(line_phases)
            phases = "abc"
            n_phases = self.dss.CktElement.NumPhases()
            if n_phases < 3:
                active_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]
                active_phases = np.array(active_phases).astype(int) - 1
                phases = "".join("abc"[i] for i in active_phases)

            each_line = dict(
                fb=self.bus_names_to_index_map[bus1],
                tb=self.bus_names_to_index_map[bus2],
                raa=z_matrix_real[0, 0] / z_base,
                rab=z_matrix_real[0, 1] / z_base,
                rac=z_matrix_real[0, 2] / z_base,
                rbb=z_matrix_real[1, 1] / z_base,
                rbc=z_matrix_real[1, 2] / z_base,
                rcc=z_matrix_real[2, 2] / z_base,
                xaa=z_matrix_imag[0, 0] / z_base,
                xab=z_matrix_imag[0, 1] / z_base,
                xac=z_matrix_imag[0, 2] / z_base,
                xbb=z_matrix_imag[1, 1] / z_base,
                xbc=z_matrix_imag[1, 2] / z_base,
                xcc=z_matrix_imag[2, 2] / z_base,
                type=element_type,
                name=element_name,
                status=switch_status,
                s_base=s_base,
                v_ln_base=base_kv_ln * 1000,
                z_base=z_base,
                phases=phases,
            )
            line_data.append(each_line)
            flag = self.dss.PDElements.Next()

        # combine lines between identical buses.
        branch_df = pd.DataFrame(line_data)
        branch_df = (
            branch_df.groupby(by=["fb", "tb"], as_index=False)
            .agg(
                {
                    "fb": "max",
                    "tb": "max",
                    "raa": "sum",
                    "rab": "sum",
                    "rac": "sum",
                    "rbb": "sum",
                    "rbc": "sum",
                    "rcc": "sum",
                    "xaa": "sum",
                    "xab": "sum",
                    "xac": "sum",
                    "xbb": "sum",
                    "xbc": "sum",
                    "xcc": "sum",
                    "type": "first",
                    "name": "sum",
                    "status": "first",
                    "s_base": "first",
                    "v_ln_base": "first",
                    "z_base": "first",
                    "phases": "sum",
                }
            )
            .sort_values(by=["tb", "fb"], ignore_index=True)
            .reset_index(drop=True)
        )
        return branch_df

    def get_bus_data(self) -> pd.DataFrame:
        """Extract the bus data from the distribution model.

        Args:
            source_voltage (float, optional): Voltage of the source (for all phases) in per unit (pu). Defaults to 1.0.
            s_base (float, optional): MVA base of the system (in VA). Defaults to 1000000 (or 1 MVA).
            v_min (float, optional): minimum voltage limit of the system in pu. Defaults to 0.95.
            v_max (float, optional): maximum voltage limit of the system in pu. Defaults to 1.05.
            cvr_p (float, optional): conservative voltage reduction parameter for p (0 means no voltage dependence). Defaults to 0.
            cvr_q (float, optional): conservative voltage reduction parameter for q (0 means no voltage dependence). Defaults to 0.

        Returns:
            pd.DataFrame: bus data in DataFrame format
        """
        source_voltage = self.dss.Vsources.PU()
        s_base = self.s_base
        v_min = self.v_min
        v_max = self.v_max
        cvr_p = self.cvr_p
        cvr_q = self.cvr_q
        all_buses_names = self.dss.Circuit.AllBusNames()
        # all_loads = self.get_loads()
        load_df = self._get_loads()
        bus_data = []
        for bus_id, bus in enumerate(all_buses_names):
            # need to set the nodes active before extracting their info
            self.dss.Circuit.SetActiveBus(bus)
            bus_type = "PQ"
            v = 1
            if (
                len(self.dss.Bus.AllPCEatBus()) > 0
                and "Vsource" in self.dss.Bus.AllPCEatBus()[0]
            ):
                v = source_voltage
                bus_type = "SWING"
            active_bus_name = self.dss.Bus.Name()
            v_ln_base = self.dss.Bus.kVBase() * 1000
            each_bus = dict(
                id=self.bus_names_to_index_map[bus],  # bus id for each active bus
                name=active_bus_name,  # name of the active bus
                bus_type=bus_type,  # SWING if source else PQ
                v_a=v,  # p.u. voltage of the active bus in phase a
                v_b=v,  # p.u. voltage of the active bus in phase b
                v_c=v,  # p.u. voltage of the active bus in phase c
                v_ln_base=v_ln_base,  # line-to-phase voltage base of the active bus
                s_base=s_base,  # s_base of the system
                v_min=v_min,  # minimum p.u. voltage for the bus
                v_max=v_max,  # maximum p.u. voltage for the bus
                cvr_p=cvr_p,  # conservative voltage reduction parameter for active power
                cvr_q=cvr_q,  # conservative voltage reduction parameter for reactive power
                phases=self.num_phase_map[
                    str(self.dss.Bus.Nodes())
                ],  # bus phases a,b,c
                has_gen=(
                    True if active_bus_name in self.gen_buses else False
                ),  # if the bus has a generator or not
                has_load=(
                    True if active_bus_name in self.load_buses else False
                ),  # if the bus has a load or not
                has_cap=(
                    True if active_bus_name in self.cap_buses else False
                ),  # if the bus has a capacitor or not
                # be careful that X gives you lon and Y gives you lat
                # extra elements
                latitude=self.dss.Bus.Y(),  # latitude of the bus location (Y)
                longitude=self.dss.Bus.X(),
            )  # longitude of the bus location (X)
            bus_data.append(each_bus)
        bus_df = pd.DataFrame(bus_data)
        bus_df = (
            pd.merge(load_df, bus_df, on=["id"], how="outer")
            .sort_values(by="id", ignore_index=True)
            .fillna(0)
        )
        return bus_df

    def get_gen_data(self) -> pd.DataFrame:
        s_base = self.s_base
        generator_flag = self.dss.Generators.First()
        gen_data = []

        while generator_flag:
            bus_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]
            n_phases = len(bus_phases)
            if len(bus_phases) == 0 or len(bus_phases) >= 3:
                n_phases = 3
            active_phases = np.array([0, 1, 2])
            if n_phases < 3:
                active_phases = (
                    np.array(self.dss.CktElement.BusNames()[0].split(".")[1:]).astype(
                        int
                    )
                    - 1
                )

            active_power_per_phase = (
                self.dss.Generators.kW() / n_phases * 1000
            ) / s_base
            reactive_power_per_phase = (
                self.dss.Generators.kvar() / n_phases * 1000
            ) / s_base
            apparent_power_rating = (
                self.dss.Generators.kVARated() / n_phases * 1000 / s_base
            )
            gen_name = self.dss.Generators.Name()
            bus_name = self.dss.Generators.Bus1().split(".")[0]
            each_gen = dict(
                id=self.bus_names_to_index_map[bus_name],
                name=gen_name,
            )
            phases = ""
            for phase_id in active_phases:
                ph = "abc"[phase_id]
                each_gen[f"p{ph}"] = active_power_per_phase
                phases = phases + ph
            for phase_id in active_phases:
                ph = "abc"[phase_id]
                each_gen[f"q{ph}"] = reactive_power_per_phase
            for phase_id in active_phases:
                ph = "abc"[phase_id]
                each_gen[f"s{ph}_max"] = apparent_power_rating
            for ph in "abc":
                if ph not in phases:
                    each_gen[f"p{ph}"] = 0
                    each_gen[f"q{ph}"] = 0
                    each_gen[f"s{ph}_max"] = 0

            each_gen["phases"] = phases

            each_gen["qa_max"] = (None,)
            each_gen["qb_max"] = (None,)
            each_gen["qc_max"] = (None,)
            each_gen["qa_min"] = (None,)
            each_gen["qb_min"] = (None,)
            each_gen["qc_min"] = (None,)

            gen_data.append(each_gen)
            generator_flag = self.dss.Generators.Next()

        pv_flag = self.dss.PVsystems.First()
        while pv_flag:
            bus_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]
            n_phases = len(bus_phases)
            if len(bus_phases) == 0 or len(bus_phases) >= 3:
                n_phases = 3
            active_phases = np.array([0, 1, 2])
            if n_phases < 3:
                active_phases = (
                    np.array(self.dss.CktElement.BusNames()[0].split(".")[1:]).astype(
                        int
                    )
                    - 1
                )

            active_power_per_phase = (
                self.dss.PVsystems.Pmpp() / n_phases * 1000
            ) / s_base
            reactive_power_per_phase = (
                self.dss.PVsystems.kvar() / n_phases * 1000
            ) / s_base
            apparent_power_rating = (
                self.dss.PVsystems.kVARated() / n_phases * 1000 / s_base
            )
            gen_name = self.dss.PVsystems.Name()
            bus_name = self.dss.CktElement.BusNames()[0].split(".")[0]
            each_gen = dict(
                id=self.bus_names_to_index_map[bus_name],
                name=gen_name,
            )
            phases = ""
            for phase_id in active_phases:
                ph = "abc"[phase_id]
                each_gen[f"p{ph}"] = active_power_per_phase
                phases = phases + ph
            for phase_id in active_phases:
                ph = "abc"[phase_id]
                each_gen[f"q{ph}"] = reactive_power_per_phase
            for phase_id in active_phases:
                ph = "abc"[phase_id]
                each_gen[f"s{ph}_max"] = apparent_power_rating
            for ph in "abc":
                if ph not in phases:
                    each_gen[f"p{ph}"] = 0
                    each_gen[f"q{ph}"] = 0
                    each_gen[f"s{ph}_max"] = 0

            each_gen["phases"] = phases

            each_gen["qa_max"] = (None,)
            each_gen["qb_max"] = (None,)
            each_gen["qc_max"] = (None,)
            each_gen["qa_min"] = (None,)
            each_gen["qb_min"] = (None,)
            each_gen["qc_min"] = (None,)

            gen_data.append(each_gen)
            pv_flag = self.dss.PVsystems.Next()

        gen_df = pd.DataFrame(gen_data)
        if len(gen_data) < 1:
            gen_df = pd.DataFrame(
                {
                    "id": [0],
                    "name": ["none"],
                    "pa": [0],
                    "pb": [0],
                    "pc": [0],
                    "qa": [0],
                    "qb": [0],
                    "qc": [0],
                    "sa_max": [0],
                    "sb_max": [0],
                    "sc_max": [0],
                    "phases": ["abc"],
                    "qa_max": [0],
                    "qb_max": [0],
                    "qc_max": [0],
                    "qa_min": [0],
                    "qb_min": [0],
                    "qc_min": [0],
                }
            )
        gen_df = gen_df.groupby(by=["id"], as_index=False).agg(
            dict(
                id="first",
                name="first",
                pa="sum",
                pb="sum",
                pc="sum",
                qa="sum",
                qb="sum",
                qc="sum",
                sa_max="sum",
                sb_max="sum",
                sc_max="sum",
                phases="sum",
                qa_max="sum",
                qb_max="sum",
                qc_max="sum",
                qa_min="sum",
                qb_min="sum",
                qc_min="sum",
            )
        )
        return gen_df

    def get_cap_data(self) -> pd.DataFrame:
        s_base = self.s_base
        flag = self.dss.Capacitors.First()
        cap_data = []
        while flag:
            cap_bus_name = self.dss.CktElement.BusNames()[0].split(".")[0]
            cap_bus_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]

            # convert this to string to be consistent with how we conver num to phase letters
            cap_bus_phases = str([int(phase) for phase in cap_bus_phases])
            if cap_bus_phases == "[]":
                # three phases are usually represented by either .1.2.3 or nothing in opendss
                # for second case we should ensure that 3 phase is actually represented
                cap_bus_phases = "[1, 2, 3]"

            cap_phase = self.num_phase_map[cap_bus_phases]

            if cap_phase != "abc":
                each_cap = dict(
                    id=self.bus_names_to_index_map[cap_bus_name],
                    name=cap_bus_name,
                    qa=(
                        self.dss.Capacitors.kvar() * 1000 / s_base
                        if cap_phase in {"a"}
                        else 0
                    ),
                    qb=(
                        self.dss.Capacitors.kvar() * 1000 / s_base
                        if cap_phase in {"b"}
                        else 0
                    ),
                    qc=(
                        self.dss.Capacitors.kvar() * 1000 / s_base
                        if cap_phase in {"c"}
                        else 0
                    ),
                    phases=cap_phase,
                )
            else:
                each_cap = dict(
                    id=self.bus_names_to_index_map[cap_bus_name],
                    name=cap_bus_name,
                    qa=(self.dss.Capacitors.kvar() * 1000 / 3) / s_base,
                    qb=(self.dss.Capacitors.kvar() * 1000 / 3) / s_base,
                    qc=(self.dss.Capacitors.kvar() * 1000 / 3) / s_base,
                    phases=cap_phase,
                )

            cap_data.append(each_cap)
            flag = self.dss.Capacitors.Next()
        cap_df = pd.DataFrame(cap_data)
        if len(cap_data) < 1:
            cap_df = pd.DataFrame(
                {
                    "id": [0],
                    "name": [0],
                    "qa": [0],
                    "qb": [0],
                    "qc": [0],
                    "phases": ["abc"],
                }
            )

        cap_df = cap_df.groupby(by=["id"], as_index=False).agg(
            dict(
                id="first",
                name="first",
                qa="sum",
                qb="sum",
                qc="sum",
                phases="sum",
            )
        )
        return cap_df

    def get_reg_data(self) -> pd.DataFrame:
        s_base = self.s_base
        flag = self.dss.Transformers.First()
        reg_data = []
        while flag:
            switch_status = None
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            element_name = self.dss.CktElement.Name().lower().split(".")[1]
            z_matrix_real = z_matrix_imag = np.zeros((3, 3))
            if element_type not in ["transformer"]:
                flag = self.dss.PDElements.Next()
                continue
            bus1 = self.dss.CktElement.BusNames()[0].split(".")[0]
            bus2 = self.dss.CktElement.BusNames()[1].split(".")[0]
            self.dss.Circuit.SetActiveBus(bus2)
            # self.dss.Circuit.SetActiveBus(self.dss.Lines.Bus2().split(".")[0])
            base_kv_ln = self.dss.Bus.kVBase()
            z_base = (base_kv_ln * 1000) ** 2 / s_base
            line_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]
            line_phases = sorted(line_phases)

            # convert this to string to be consistent with how we conver num to phase letters
            line_phases = str([int(phase) for phase in line_phases])
            if line_phases == "[]":
                # three phases are usually represented by either .1.2.3 or nothing in opendss
                # for second case we should ensure that 3 phase is actually represented
                line_phases = "[1, 2, 3]"
            try:

                line_phase = self.num_phase_map[line_phases]
            except:
                breakpoint()
            tap = self.dss.Transformers.Tap()
            each_reg = {}
            each_reg["fb"] = self.bus_names_to_index_map[bus1]
            each_reg["tb"] = self.bus_names_to_index_map[bus2]
            for ph in line_phase:
                each_reg[f"ratio_{ph}"] = tap
            each_reg["phases"] = line_phase
            reg_data.append(each_reg)

            flag = self.dss.Transformers.Next()

        # combine lines between identical buses.
        reg_df = pd.DataFrame(reg_data)
        if len(reg_data) < 1:
            reg_df = pd.DataFrame(
                {
                    "fb": [0],
                    "tb": [0],
                    "ratio_a": [0],
                    "ratio_b": [0],
                    "ratio_c": [0],
                    "phases": ["abc"],
                }
            )
        reg_df = reg_df.groupby(["fb", "tb"]).agg(
            {
                "fb": "first",
                "tb": "first",
                "ratio_a": "max",
                "ratio_b": "max",
                "ratio_c": "max",
                "phases": "sum",
            }
        )
        reg_df = reg_df.reset_index(drop=True)
        reg_df = reg_df.sort_values(by="tb", ignore_index=True).fillna(1)
        reg_df["tap_a"] = (reg_df["ratio_a"] - 1) / 0.00625
        reg_df["tap_b"] = (reg_df["ratio_b"] - 1) / 0.00625
        reg_df["tap_c"] = (reg_df["ratio_c"] - 1) / 0.00625
        return reg_df

    def _get_loads(self) -> dict[str, list[float]]:
        """Extract load information for each node for each phase. This method extracts load on the exact bus(node) as
        modeled in the distribution model, including secondary.

        Returns:
            load_per_phase(pd.DataFrame): Per phase load data in a pandas dataframe
        """
        s_base = self.s_base
        load_df = pd.DataFrame(
            [], columns=["id", "name", "pl_a", "ql_a", "pl_b", "ql_b", "pl_c", "ql_c"]
        )
        loads_flag = self.dss.Loads.First()
        load_data = []
        while loads_flag:
            connected_buses = self.dss.CktElement.BusNames()
            if len(connected_buses) > 1:
                raise Exception("Multiple connected buses")
            bus = connected_buses[0]
            bus_name = bus.split(".")[0]
            each_load = {
                "id": 0,
                "pl_a": 0,
                "ql_a": 0,
                "pl_b": 0,
                "ql_b": 0,
                "pl_c": 0,
                "ql_c": 0,
            }
            bus_split = bus.split(".")
            each_load["id"] = self.bus_names_to_index_map[bus_name]
            connected_phase_secondary = bus_split[1:]

            # conductor power contains info on active and reactive power
            conductor_power = np.array(self.dss.CktElement.Powers())
            # nonzero_power_indices = np.where(conductor_power != 0)[0]
            # nonzero_power = conductor_power[nonzero_power_indices]
            # Extract P and Q values (every alternate elements)
            a1 = np.exp(-1 / 6 * 1j * np.pi)
            a2 = np.exp(-5 / 6 * 1j * np.pi)
            p_values = conductor_power[::2]
            q_values = conductor_power[1::2]
            phases = "abc"
            n_phases = self.dss.Loads.Phases()
            pf = self.dss.Loads.PF()
            kw = self.dss.Loads.kW()
            kv = self.dss.Loads.kV()
            kvar = self.dss.Loads.kvar()
            is_delta = self.dss.Loads.IsDelta()
            if len(connected_phase_secondary) > 0:
                phases = "".join("abc"[int(n) - 1] for n in connected_phase_secondary)
            # if len(phases) != n_phases:
            #     raise Exception("Number of load phases does not match with bus phases")
            for phase_index, ph in enumerate(phases):
                each_load[f"pl_{ph}"] = p_values[phase_index] * 1000 / s_base
                each_load[f"ql_{ph}"] = q_values[phase_index] * 1000 / s_base
            load_data.append(each_load)
            loads_flag = self.dss.Loads.Next()
        load_df = pd.DataFrame(load_data)
        load_df = load_df.groupby("id").agg(
            {
                "id": "first",
                "pl_a": "sum",
                "ql_a": "sum",
                "pl_b": "sum",
                "ql_b": "sum",
                "pl_c": "sum",
                "ql_c": "sum",
            }
        )
        load_df = load_df.fillna(0)
        return load_df.reset_index(drop=True)

    def to_csv(self, dir_name: str = None, overwrite: bool = True) -> None:

        if dir_name is None:
            dir_name = "testfiles"

        Path(dir_name).mkdir(parents=True, exist_ok=overwrite)
        self.branch_data.to_csv(f"{dir_name}/branch_data.csv", index=False)
        self.bus_data.to_csv(f"{dir_name}/bus_data.csv", index=False)
        self.cap_data.to_csv(f"{dir_name}/cap_data.csv", index=False)
        self.gen_data.to_csv(f"{dir_name}/gen_data.csv", index=False)
        self.reg_data.to_csv(f"{dir_name}/reg_data.csv", index=False)

    def update_gen_q(self, q: pd.DataFrame):
        flag = self.dss.Generators.First()
        while flag:
            bus_phases = np.array(
                self.dss.CktElement.BusNames()[0].split(".")[1:]
            ).astype(int)
            n_phases = len(bus_phases)
            if len(bus_phases) == 0 or len(bus_phases) >= 3:
                n_phases = 3
            active_phases = np.array([0, 1, 2])
            if n_phases < 3:
                active_phases = bus_phases - 1
            phase_columns = ["abc"[ph_idx] for ph_idx in active_phases]
            bus = self.dss.Generators.Bus1().split(".")[0]
            bus_id = self.bus_names_to_index_map[bus]
            kvar = q.loc[bus_id, phase_columns].sum() * self.s_base / 1000
            self.dss.Generators.kvar(kvar)
            flag = self.dss.Generators.Next()


def main() -> None:
    # dss_data = DSSParser(r'ieee9500_dss/Master-unbal-initial-config.dss')
    dss_data = DSSParser(r"ieee13Bus/IEEE13Nodeckt.dss")
    print(dss_data.get_branches())


if __name__ == "__main__":
    main()
