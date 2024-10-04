import json
import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from distopf.glmpy.basic import Gridlabd


# from glmpy.graph import fix_reversed_links


class Link:
    def __init__(self, el):
        self.name = el.find("./name").text
        self.id = el.find("./id").text
        self.from_node = el.find("./from_node").text
        self.to_node = el.find("./to_node").text
        self.from_voltage = float(el.find("./from_voltage").text)
        self.to_voltage = float(el.find("./to_voltage").text)
        self.phases = el.find("./phases").text
        self.a_mat = self.make_mat(el.find("./a_matrix"), "a")
        self.b_mat = self.make_mat(el.find("./b_matrix"), "b")
        self.c_mat = self.make_mat(el.find("./c_matrix"), "c")
        self.d_mat = self.make_mat(el.find("./d_matrix"), "d")
        self.A_mat = self.make_mat(el.find("./A_matrix"), "A")
        self.B_mat = self.make_mat(el.find("./B_matrix"), "B")
        self.S_base = None
        self.v_base = None  # need to determine this from network by looking for upstream transformer secondary voltage
        self.z_base = None

    @staticmethod
    def make_mat(el, name):
        mat = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                mat[i, j] = complex(el.find(f"./{name}{i + 1}{j + 1}").text)
        return mat


class Switch(Link):
    def __init__(self, sw):
        super().__init__(sw)
        self.o_type = "switch"


class Fuse(Link):
    def __init__(self, sw):
        super().__init__(sw)
        self.o_type = "fuse"


class Recloser(Link):
    def __init__(self, sw):
        super().__init__(sw)
        self.o_type = "recloser"


class Transformer(Link):
    def __init__(self, xf):
        super().__init__(xf)
        self.o_type = "transformer"
        self.xfmr_config = xf.find("./xfmr_config").text
        self.power_rating = float(xf.find("./power_rating").text)
        self.resistance = float(xf.find("./resistance").text)
        self.reactance = float(xf.find("./reactance").text)


class UndergroundLine(Link):
    def __init__(self, ul):
        self.o_type = "underground_line"
        super().__init__(ul)
        self.length = float(ul.find("./length").text)


class OverheadLine(Link):
    def __init__(self, ol):
        self.o_type = "overhead_line"
        super().__init__(ol)
        self.length = float(ol.find("./length").text)


class Regulator(Link):
    def __init__(self, reg):
        super().__init__(reg)
        self.o_type = "regulator"
        self.tapA = float(reg.find("./tapA").text)
        self.tapB = float(reg.find("./tapB").text)
        self.tapC = float(reg.find("./tapC").text)


class Glm2csv:
    def __init__(
        self,
        impedance_xml,
        main_file,
        model_path,
        out_dir,
        s_base=1e3,
        use_pu_z: bool = False,
    ):
        self.tree = ET.parse(impedance_xml)
        self.glm = Gridlabd(main_file, model_path)
        self.out_dir = Path(out_dir)
        self.s_base = s_base  # s_base is per phase
        self.use_pu_z = use_pu_z

        self.glm.remove_quotes_from_obj_names()  # having quotes in the object names messes stuff up
        # self.glm.remove_quotes_from_obj_references()  # need to fix the object references after cleaning the names
        self.root = self.tree.getroot()
        self.time = self.root.find("./Time").text
        self.link_objs = self.get_link_obj_list(self.root)
        self.swing_buses = self.glm.find_objects_with_property_value(
            "bustype", "SWING", search_types=["meter", "node"], prepend_class=True
        )
        self.g = nx.DiGraph()
        # find and add nodes to graph from glm
        # swing = []
        for obj_type in ["meter", "node", "load", "capacitor"]:
            if self.glm.model.get(obj_type) is not None:
                for obj_name, obj_dict in self.glm.model[obj_type].items():
                    if self.glm.model[obj_type][obj_name].get("parent") is None:
                        self.g.add_node(
                            obj_type + ":" + obj_name.strip('"').strip("'"),
                            ob_dict=obj_dict,
                        )
                    # if self.glm.model[obj_type][obj_name].get('bustype') == 'SWING':
                    #     swing.append(obj_type + ':' + obj_name.strip('\"').strip('\''))
        # Prepare and add edges to graph
        self.g_closed = self.g.copy()
        link_types = [  # list of link-like object types in GridLAB-D
            "link",
            "overhead_line",
            "underground_line",
            "triplex_line",
            "transformer",
            "regulator",
            "fuse",
            "switch",
            "recloser",
            "relay",
            "sectionalizer",
            "series_reactor",
        ]
        transformer_edges = (
            []
        )  # this list will be useful for determining base voltage levels later
        for link_type in link_types:  # add edges to graph based from GLM
            if self.glm.model.get(link_type) is not None:
                for link_name, link_dict in self.glm.model[link_type].items():
                    from_obj_class = self.glm.get_object_type(link_dict["from"])
                    to_obj_class = self.glm.get_object_type(link_dict["to"])
                    from_node_str = (
                        from_obj_class + ":" + link_dict["from"].strip('"').strip("'")
                    )
                    to_node_str = (
                        to_obj_class + ":" + link_dict["to"].strip('"').strip("'")
                    )
                    assert from_node_str in self.g.nodes
                    assert to_node_str in self.g.nodes
                    self.g.add_edge(
                        from_node_str,
                        to_node_str,
                        ob_dict=link_dict,
                        name=link_name,
                        ob_type=link_type,
                    )
                    # create list of transformer edges
                    if link_type == "transformer":
                        transformer_edges.append((from_node_str, to_node_str))
                    # also create graph with no open switches
                    if link_type in ["switch", "recloser", "sectionalizer", "fuse"]:
                        if link_dict["status"] == "CLOSED":
                            self.g_closed.add_edge(
                                from_node_str,
                                to_node_str,
                                ob_dict=link_dict,
                                name=link_name,
                                ob_type=link_type,
                            )
                    else:
                        self.g_closed.add_edge(
                            from_node_str,
                            to_node_str,
                            ob_dict=link_dict,
                            name=link_name,
                            ob_type=link_type,
                        )

        # Add impedance values from xml data
        for link in self.link_objs:
            if self.g.get_edge_data(link.from_node, link.to_node) is not None:
                self.g.get_edge_data(link.from_node, link.to_node)["ob"] = link
                # self.g_closed.get_edge_data(link.from_node, link.to_node)['ob'] = link
            elif self.g.get_edge_data(link.to_node, link.from_node) is not None:
                self.g.get_edge_data(link.to_node, link.from_node)["ob"] = link
                # self.g_closed.get_edge_data(link.to_node, link.from_node)['ob'] = link
            else:
                raise Exception(
                    f"link not found between {link.from_node} and {link.to_node}"
                )
            # self.g_.add_edge(link.from_node.split(':')[-1], link.to_node.split(':')[-1], ob=link)
        # Assign base voltages here.
        # 1. Assign base voltages to nodes on either side of each transformer
        voltage_region_sets = {}
        if "transformer" in self.glm.model.keys():
            g_without_xfmrs = nx.Graph(self.g.copy())
            g_without_xfmrs.remove_edges_from(transformer_edges)
            transformer_split_sets = list(nx.connected_components(g_without_xfmrs))
            for xfmr_name in self.glm.model["transformer"].keys():
                from_bus = self.glm.model["transformer"][xfmr_name].get("from")
                from_bus_class = self.glm.get_object_type(from_bus, ["meter", "node"])
                from_bus = from_bus_class + ":" + from_bus
                to_bus = self.glm.model["transformer"][xfmr_name].get("to")
                to_bus_class = self.glm.get_object_type(to_bus, ["meter", "node"])
                to_bus = to_bus_class + ":" + to_bus
                xfmr_config_name = self.glm.model["transformer"][xfmr_name].get(
                    "configuration"
                )
                xfmr_config = self.glm.model["transformer_configuration"].get(
                    xfmr_config_name
                )
                v_line_base_pri = float(xfmr_config.get("primary_voltage"))
                v_line_base_sec = float(xfmr_config.get("secondary_voltage"))
                # 1. Assign base voltages to nodes on either side of each transformer
                self.g.nodes[from_bus]["v_ll_base"] = v_line_base_pri
                self.g.nodes[to_bus]["v_ll_base"] = v_line_base_sec
                # 2. Assign transformers the secondary voltage as their base
                if self.g.get_edge_data(from_bus, to_bus) is not None:
                    self.g.get_edge_data(from_bus, to_bus)[
                        "v_ll_base"
                    ] = v_line_base_sec
                # 3. Assign voltages to sets of nodes
                primary_set = [
                    node_set
                    for node_set in transformer_split_sets
                    if from_bus in node_set
                ][0]
                secondary_set = [
                    node_set
                    for node_set in transformer_split_sets
                    if to_bus in node_set
                ][0]
                if voltage_region_sets.get(v_line_base_pri) is None:
                    voltage_region_sets[v_line_base_pri] = primary_set
                else:
                    voltage_region_sets[v_line_base_pri] = (
                        voltage_region_sets[v_line_base_pri] | primary_set
                    )
                if voltage_region_sets.get(v_line_base_sec) is None:
                    voltage_region_sets[v_line_base_sec] = secondary_set
                else:
                    voltage_region_sets[v_line_base_sec] = (
                        voltage_region_sets[v_line_base_sec] | secondary_set
                    )
        else:
            v_ln_nom = float(
                self.glm.get_object_property_value(
                    self.swing_buses[0], "nominal_voltage"
                )
            )
            v_ll = v_ln_nom * np.sqrt(3)
            voltage_region_sets[v_ll] = self.g.nodes()
        self.voltage_region_sets = voltage_region_sets

        # Trim nodes not connected to main system.
        # Assume larges connected component is the system and all swing nodes are there
        components = nx.weakly_connected_components(self.g)
        components_closed = nx.weakly_connected_components(self.g_closed)
        # self.swing_buses = \
        #     self.glm.find_obj_by_prop('bustype', 'SWING', class_list=['meter', 'node'], prepend_class=True)
        largest_set = max(components)
        largest_set_closed = max(components_closed)
        self.g_complete = self.g.copy()
        self.g_closed_complete = self.g_closed.copy()
        self.g = self.g_complete.subgraph(largest_set)
        self.g_closed = self.g_closed_complete.subgraph(largest_set_closed)
        # self.g_ = max(components_)
        # ~~~~ Relabel nodes ~~~~~~~~~
        self.swing_bus = None
        for bus in self.swing_buses:
            if bus in self.g:
                self.swing_bus = bus
                break

        # reverse backwards links
        self.g = self.fix_reversed_links(self.g, self.swing_bus)
        # self.swing_bus = self.find_swing()
        node_list = nx.dfs_preorder_nodes(nx.Graph(self.g), self.swing_bus)
        self.node_map2name = {}
        self.node_map2number = {}
        i = 0
        for i, node_name in enumerate(node_list):
            self.node_map2name[i + 1] = node_name.strip('"').strip("'")
            self.node_map2number[node_name.strip('"').strip("'")] = i + 1
        self.n = i + 1
        with open(self.out_dir / "node_map.json", "w") as f:
            json.dump(self.node_map2number, f, indent=4)
        self.create_powerdata_csv()
        self.create_branch_csv()

    def create_branch_csv(self):
        # Parameters used:
        # self.g
        # self.swing_bus
        # self.node_map2number
        # self.voltage_region_sets
        # self.use_pu_z
        # self.s_base
        # self.out_dir
        # ~~~~ Create Branch Data ~~~~
        names = [
            "fb",
            "tb",
            "raa",
            "rab",
            "rac",
            "rbb",
            "rbc",
            "rcc",
            "xaa",
            "xab",
            "xac",
            "xbb",
            "xbc",
            "xcc",
            "type",
            "name",
            "status",
            "s_base",
            "v_ln_base",
            "z_base",
        ]
        default_switch_z = np.array(
            [0.0001, 0, 0, 0.0001, 0, 0.0001, -0.0001, 0, 0, -0.0001, 0, -0.0001]
        )
        branch = pd.DataFrame(columns=names)
        # print(branch)

        # edges = np.array(list(nx.dfs_edges(self.g, source=self.swing_bus)))
        # assert(len(list(nx.dfs_edges(self.g, source=self.swing_bus))) == len(self.g.edges()))
        print(len(list(nx.dfs_edges(self.g, source=self.swing_bus))))
        print(len(self.g.edges()))
        edges = self.g.edges()
        for edge in edges:
            r = []
            x = []
            fb = int(self.node_map2number[edge[0]])
            tb = int(self.node_map2number[edge[1]])
            ob = self.g.get_edge_data(edge[0], edge[1]).get("ob")
            ob_dict = self.g.get_edge_data(edge[0], edge[1]).get("ob_dict")
            ob_type = self.g.get_edge_data(edge[0], edge[1]).get("ob_type")
            ob_name = self.g.get_edge_data(edge[0], edge[1]).get("name")
            phases = ob_dict.get("phases")
            # get base voltage using downstream bus (edge[1])
            v_ln_base = None

            for v_ll_base, node_set in self.voltage_region_sets.items():
                if edge[1] in node_set:
                    v_ln_base = v_ll_base / np.sqrt(3)
            if v_ln_base is None:
                raise ValueError(f"v_ln_base was not found for {ob_name}")
            if self.use_pu_z:
                z_base = v_ln_base**2 / self.s_base  # s_base is per phase
            else:
                z_base = 1
            row = [fb, tb]
            # if ob is None:
            #     row.extend(default_switch_z / z_base)  # switch impedance
            # else:
            for i in range(3):
                for j in range(i, 3):
                    _r = 0
                    _x = 0
                    if ob is not None:
                        _r = ob.b_mat[i, j].real
                        _x = ob.b_mat[i, j].imag
                    # some open reclosers may show up in the xml with zero impedance
                    if i == 0 and j == 0 and _r == 0 and _x == 0 and "A" in phases:
                        _r = 0.0001
                        _x = -0.0001
                    if i == 1 and j == 1 and _r == 0 and _x == 0 and "B" in phases:
                        _r = 0.0001
                        _x = -0.0001
                    if i == 2 and j == 2 and _r == 0 and _x == 0 and "C" in phases:
                        _r = 0.0001
                        _x = -0.0001

                    r.append(_r / z_base)
                    x.append(_x / z_base)
                    # r.append(ob.b_mat[i, j].real / z_base)
                    # x.append(ob.b_mat[i, j].imag / z_base)
            row.extend(r)
            row.extend(x)
            row.append(ob_type)
            row.append(ob_name)
            row.append(ob_dict.get("status"))
            row.append(self.s_base)
            row.append(v_ln_base)
            row.append(z_base)
            # print(row)
            branch.loc[len(branch.index)] = row
        branch = branch.sort_values("tb")
        branch.to_csv(
            self.out_dir / "branchdata.csv", sep=",", header=True, index=False
        )
        # print(edges)

    def create_powerdata_csv(self):
        bus_col_names = [
            "id",
            "Pa",
            "Qa",
            "Pb",
            "Qb",
            "Pc",
            "Qc",
            "CapA",
            "CapB",
            "CapC",
            "PgA",
            "PgB",
            "PgC",
            "name",
            "bus_type",
            "Vln",
            "v_ln_base",
            "s_base",
        ]
        data = np.zeros((self.n, len(bus_col_names)))
        data[:, 0] = range(1, self.n + 1)
        bus_data = pd.DataFrame(data=data, columns=bus_col_names)
        for nid, node_name in self.node_map2name.items():
            bus_data.loc[nid - 1, "name"] = node_name.split(":")[-1]
            bus_data.loc[nid - 1, "Vln"] = self.glm.get_object_property_value(
                node_name, "nominal_voltage"
            )
            v_ln_base = None
            for v_ll_base, node_set in self.voltage_region_sets.items():
                if node_name in node_set:
                    v_ln_base = v_ll_base / np.sqrt(3)
            if v_ln_base is None:
                raise ValueError(f"v_ln_base was not found for {node_name}")
            bus_data.loc[nid - 1, "v_ln_base"] = v_ln_base
            bus_data.loc[nid - 1, "s_base"] = self.s_base
            # mark if it is a swing bus
            if node_name in self.swing_buses:
                bus_data.loc[nid - 1, "bus_type"] = "SWING"
            else:
                bus_data.loc[nid - 1, "bus_type"] = "PQ"

        # connect loads, generators, capacitors, etc to parent nodes/meters

        # get loads:
        # parse loads as constant p and q loads on each phase
        for load_name, load in self.glm.model["load"].items():
            # parent = load.get('parent')
            parent, parent_class = self.glm.get_final_parent(load_name, "load")
            # print(load_name, parent)
            parent = parent.strip('"').strip("'")
            load_name = load_name.strip('"').strip("'")
            if parent is None:
                nid = self.node_map2number[f"load:{load_name}"]
            else:
                nid = self.node_map2number.get(f"{parent_class}:{parent}")
            if nid is None:
                break

            # Constant PQ Delta Connected Loads
            # must assume balanced voltage
            sab = complex(load.get("constant_power_AB", complex(0)))
            sbc = complex(load.get("constant_power_BC", complex(0)))
            sca = complex(load.get("constant_power_CA", complex(0)))
            s_delta = np.array([sab, sbc, sca])
            s = np.zeros(3, dtype=complex)
            sa = sb = sc = complex(0)
            minus30 = np.exp(-1j * np.pi / 6)  # 30 deg
            minus120 = np.exp(-1j * np.pi * 2 / 3)  # -120 deg
            delta_to_wye = (
                minus30
                / np.sqrt(3)
                * np.array(
                    [
                        [1, 0, -1 * minus120],
                        [-1 * minus120, 1, 0],
                        [0, -1 * minus120, 1],
                    ]
                )
            )
            s += delta_to_wye @ s_delta
            # rt3 = np.sqrt(3)
            # s[0] += (1/(rt3*plus30)*(sab - sca))
            # s[1] += (1/(rt3*plus30)*(-sab + sbc))
            # s[2] += (1/(rt3*plus30)*(-sbc + sca))

            # Constant PQ WYE Connected LOAD:
            s[0] += complex(load.get("constant_power_A", complex(0)))
            s[1] += complex(load.get("constant_power_B", complex(0)))
            s[2] += complex(load.get("constant_power_C", complex(0)))
            s[0] += complex(load.get("constant_power_AN", complex(0)))
            s[1] += complex(load.get("constant_power_BN", complex(0)))
            s[2] += complex(load.get("constant_power_CN", complex(0)))

            for i, ph in enumerate("ABC"):
                p = 0
                q = 0
                # Y connected Loads
                # if f'constant_power_{ph}N' in load.keys() or f'constant_power_{ph}' in load.keys():
                #     power_str = load.get(f'constant_power_{ph}N')
                #     if power_str is None:
                #         power_str = load.get(f'constant_power_{ph}')
                #     p = p + complex(power_str).real
                #     q = q + complex(power_str).imag

                # ZIP Loads
                if f"base_power_{ph}" in load.keys():
                    base_power = float(load.get(f"base_power_{ph}"))
                    power_pf = float(load.get(f"power_pf_{ph}", 1))
                    current_pf = float(load.get(f"current_pf_{ph}", 1))
                    impedance_pf = float(load.get(f"impedance_pf_{ph}", 1))
                    power_fraction = float(load.get(f"power_fraction_{ph}", 1))
                    current_fraction = float(load.get(f"current_fraction_{ph}", 0))
                    impedance_fraction = float(load.get(f"impedance_fraction_{ph}", 0))
                    p = base_power * (
                        power_fraction * power_pf
                        + current_fraction * current_pf
                        + impedance_fraction * impedance_pf
                    )
                    q = base_power * (
                        power_fraction * np.sin(np.arccos(power_pf))
                        + current_fraction * np.sin(np.arccos(current_pf))
                        + impedance_fraction * np.sin(np.arccos(impedance_pf))
                    )
                    s[i] += p + 1j * q

                bus_data.loc[nid - 1, f"P{ph.lower()}"] += s[i].real / self.s_base
                bus_data.loc[nid - 1, f"Q{ph.lower()}"] += s[i].imag / self.s_base

        # get inverters:
        if self.glm.model.get("inverter") is not None:
            for inv_name, inverter in self.glm.model["inverter"].items():
                # parent = inverter.get('inverter')
                parent, parent_class = self.glm.get_final_parent(inv_name, "inverter")
                # print(inv_name, parent)
                nid = self.node_map2number.get(f"node:{parent}")
                if nid is None:
                    nid = self.node_map2number.get(f"meter:{parent}")
                    if nid is None:
                        raise Exception(
                            f"could not find id of node for connected inverter {inv_name}"
                        )
                phases = inverter.get("phases", "ABC").upper().replace("N", "")
                p_out = float(inverter.get("P_Out"))
                for ph in phases:
                    bus_data.loc[nid - 1, f"Pg{ph}"] = (p_out / self.s_base) / len(
                        phases
                    )

        # get diesel_dg:
        if self.glm.model.get("diesel_dg") is not None:
            for dg_name, dg in self.glm.model["diesel_dg"].items():
                # parent = inverter.get('inverter')
                parent, parent_class = self.glm.get_final_parent(dg_name, "diesel_dg")
                # print(inv_name, parent)
                nid = self.node_map2number.get(f"node:{parent}")
                if nid is None:
                    nid = self.node_map2number.get(f"meter:{parent}")
                    if nid is None:
                        raise Exception(
                            f"could not find id of node for connected diesel_dg {dg_name}"
                        )
                phases = dg.get("phases", "ABC").upper().replace("N", "")
                Rated_VA = float(dg.get("Rated_VA"))
                for ph in phases:
                    power_out_ph = complex(dg.get(f"power_out_{ph}"))
                    bus_data.loc[nid - 1, f"Pg{ph}"] = (
                        (power_out_ph.real) / self.s_base
                    ) / len(phases)
        bus_data.to_csv(
            self.out_dir / "powerdata.csv", sep=",", header=True, index=False
        )

    @staticmethod
    def get_link_obj_list(impedance_xml_root):
        root = impedance_xml_root
        o_switches = []
        switches = root.findall("./switch")
        for sw in switches:
            o_switches.append(Switch(sw))

        o_reclosers = []
        reclosers = root.findall("./recloser")
        for rc in reclosers:
            o_reclosers.append(Recloser(rc))

        o_fuses = []
        fuses = root.findall("./fuse")
        for fs in fuses:
            o_fuses.append(Fuse(fs))

        o_transformers = []
        transformers = root.findall("./transformer")
        for xf in transformers:
            o_transformers.append(Transformer(xf))

        o_underground_lines = []
        underground_lines = root.findall("./underground_line")
        for ul in underground_lines:
            o_underground_lines.append(UndergroundLine(ul))

        o_overhead_lines = []
        overhead_lines = root.findall("./overhead_line")
        for ol in overhead_lines:
            o_overhead_lines.append(OverheadLine(ol))

        o_regulators = []
        regulators = root.findall("./regulator")
        for reg in regulators:
            o_regulators.append(Regulator(reg))

        objs = []
        objs.extend(o_switches)
        objs.extend(o_overhead_lines)
        objs.extend(o_underground_lines)
        objs.extend(o_transformers)
        objs.extend(o_fuses)
        objs.extend(o_reclosers)
        objs.extend(o_regulators)
        return objs

    def fix_reversed_links_bad(self, g, source):
        # find and reverse links pointing the wrong direction
        digraph = nx.DiGraph()
        if source not in g:  # make sure the source is actually a part of this graph
            raise ValueError(f"The slack bus: {source} is not in the graph.")
        for edge in self.g_closed.edges():
            # print(edge)
            if edge[0] == source:
                d0 = 0
            else:
                # d0 = nx.resistance_distance(nx.Graph(digraph), edge[0], source)
                d0 = nx.shortest_path_length(nx.Graph(self.g_closed), edge[0], source)
            if edge[1] == source:
                d1 = 0
            else:
                # d1 = nx.resistance_distance(nx.Graph(digraph), edge[1], source)
                d1 = nx.shortest_path_length(nx.Graph(self.g_closed), edge[1], source)
            if d0 <= d1:  # Direction is correct. Don't reverse.
                digraph.add_edge(edge[0], edge[1], **g.get_edge_data(edge[0], edge[1]))
            else:  # Direction is reversed. Reverse it.
                digraph.add_edge(edge[1], edge[0], **g.get_edge_data(edge[0], edge[1]))
        return digraph

    def fix_reversed_links(self, G, source):
        # G = G.copy()
        digraph = nx.DiGraph()
        undigraph = nx.Graph(self.g)
        undigraph_closed = nx.Graph(self.g_closed)

        if (
            source not in self.g
        ):  # make sure the source is actually a part of this graph
            raise ValueError(f"The slack bus: {source} is not in the graph.")

        # fixed_edges = [edge for edge in nx.dfs_edges(nx.Graph(G), source=source)]
        # edge_attr = [G.get_edge_data(edge[0], edge[1]) or G.get_edge_data(edge[1], edge[0]) for edge in fixed_edges]
        # edge_dicts = [attr[0] for attr in edge_attr]
        edge_list = []
        for edge in nx.dfs_edges(
            undigraph_closed, source
        ):  # for edge in fixed_edges:  # for edge, dict in zip(fixed_edges, edge_attr):
            digraph.add_edge(*edge, **undigraph.get_edge_data(*edge))
            # edge_list.append((edge[0], edge[1], dict))
        # digraph.add_edges_from(edge_list)
        print(len(digraph.nodes))
        missing_edges = []
        for edge in self.g.edges:
            if (
                edge not in digraph.edges
                and self.reverse_edge(edge) not in digraph.edges
            ):
                missing_edges.append((edge[0], edge[1]))
        add_edges = []
        for edge in missing_edges:
            edge_data = self.g.get_edge_data(edge[0], edge[1])
            if edge_data.get("ob_type") in ["switch", "recloser", "fuse"]:
                if edge_data.get("ob_dict") is not None:
                    if edge_data.get("ob_dict").get("status") == "OPEN":
                        digraph.add_edge(*edge, **undigraph.get_edge_data(*edge))
                        # print('OPEN')
            # if edge[0] == source:
            #     d0 = 0
            # else:
            #     # d0 = nx.resistance_distance(nx.Graph(digraph), edge[0], source)
            #     d0 = nx.shortest_path_length(undigraph, edge[0], source)
            # if edge[1] == source:
            #     d1 = 0
            # else:
            #     # d1 = nx.resistance_distance(nx.Graph(digraph), edge[1], source)
            #     d1 = nx.shortest_path_length(undigraph, edge[1], source)
            # if d0 < d1:  # Direction is correct. Don't reverse.
            #     digraph.add_edge(edge[0], edge[1], **undigraph.get_edge_data(*edge))
            # else:  # Direction is reversed. Reverse it.
            #     digraph.add_edge(edge[1], edge[0], **undigraph.get_edge_data(*edge))
        # digraph.add_edges_from(add_edges)
        # data = [G.get_edge_data(edge[0], edge[1]) or G.get_edge_data(edge[1], edge[0])  for edge in digraph.edges]
        return digraph

    @staticmethod
    def reverse_edge(edge):
        assert len(edge) >= 2
        if len(edge) == 2:
            return (edge[1], edge[0])
        if len(edge) > 2:
            return (edge[1], edge[0], *edge[2:])


if __name__ == "__main__":
    model_path_ = Path("~/PycharmProjects/CitadelsActual/gridlabd/base")
    main_file_ = model_path_ / "test_CITADEL_Feeders_Deltamode.glm"
    impedance_path_ = "xml/citadels_base.xml"
    # model_path = Path("~/PycharmProjects/dopf_trans/gridlabd/30_ders_no_helics")
    # main_file = model_path / "system.glm"
    # impedance_path = \
    #     Path("~/PycharmProjects/dopf_trans/gridlabd/30_ders_no_helics/output/impedance.xml")
    Glm2csv(impedance_path_, main_file_, model_path_, Path.cwd())
