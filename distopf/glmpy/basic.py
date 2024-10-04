import warnings
from distopf.glmpy import glmanip, graph
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from copy import deepcopy, copy

_link_types = [
    'link', 'overhead_line', 'underground_line', 'triplex_line', 'transformer',
    'regulator', 'fuse', 'switch', 'recloser', 'relay', 'sectionalizer', 'series_reactor'
]
_node_types = ['meter', 'node', 'triplex_node', 'triplex_meter', 'load', 'pqload', 'capacitor', 'recorder',
               'inverter', 'diesel_dg']


class Gridlabd:
    def __init__(self, file_path=None, base_dir_path=None):
        """

        Parameters
        ----------
        file_path
        base_dir_path
        """

        self.model = {}
        self.clock = {}
        self.directives = {}
        self.modules = {}
        self.classes = {}
        self.schedules = {}
        self.file_path = file_path
        self.base_dir_path = base_dir_path
        if file_path is not None:
            self.file_path = Path(file_path)
            if self.base_dir_path is None:
                self.base_dir_path = self.file_path.parent
            self.base_dir_path = Path(self.base_dir_path)
            self.read(self.file_path, self.base_dir_path)

    def read(self, file_path, base_dir):
        """

        Parameters
        ----------
        file_path: str or Path -- path of glm file
        base_dir: str or Path -- path of model base directory,

        Returns
        -------

        """
        self.model, self.clock, self.directives, self.modules, self.classes, self.schedules = \
            glmanip.parse(glmanip.read(file_path, base_dir))

    def write(self, filename):
        glmanip.write(filename, self.model, self.clock, self.directives, self.modules, self.classes, self.schedules)

    def swing_nodes(self):
        return self.find_objects_with_property_value('bustype', 'SWING', search_types=['meter', 'node'],
                                                      prepend_class=False)

    def find_objects_with_property_value(
            self, obj_property: str, value: str, search_types: list = None, prepend_class=False):
        """

        Parameters
        ----------
        obj_property: str
        value: str -- value of property
        search_types: list -- optional list of types to search for the object in
        prepend_class: bool -- if true, returned object names will be prepended with the object type e.g. node:node_1

        Returns
        -------
        list of object names which have the property with the given value

        """
        if search_types is None:
            search_types = self.model.keys()
        obj_list = []
        for obj_type in search_types:
            if obj_type in self.model.keys():
                for obj_name in self.model.get(obj_type):
                    if self.model[obj_type][obj_name].get(obj_property) == value:
                        if prepend_class:
                            obj_list.append(obj_type.strip("\'").strip("\"") + ':' + obj_name.strip("\'").strip("\""))
                        else:
                            obj_list.append(obj_name)
        return obj_list

    def get_object_type(self, obj_name: str, search_types: list = None):
        """

        Parameters
        ----------
        obj_name: str -- object name
        search_types: list -- optional list of types to search for the object in

        Returns
        -------
        name of the class that the object belongs to
        """
        if len(obj_name.split(":")) == 2:  # support receiving obj_name in the form class:obj_name e.g. "meter:node_2"
            class_name = obj_name.split(":")[0]
            return class_name
        if search_types is None:
            search_types = self.model.keys()
        for class_name in search_types:
            if class_name in self.model.keys():
                if obj_name in self.model[class_name].keys():
                    return class_name
        raise Warning("Did not find object class")

    def get_object_property_value(self, obj_name: str, obj_property: str, search_types: list = None):
        """

        Parameters
        ----------
        obj_name: str -- object name
        obj_property: str -- property to get the value of
        search_types: list -- optional list of types to search for the object in

        Returns
        -------
        value of property of the object
        """
        if len(obj_name.split(":")) == 2:  # support receiving obj_name in the form class:obj_name e.g. "meter:node_2"
            obj_class = obj_name.split(":")[0]
            obj_name = obj_name.split(":")[-1]
        else:
            obj_class = self.get_object_type(obj_name, search_types=search_types)
        if self.model[obj_class].get(obj_name) is None:
            return self.model[obj_class].get('\"' + obj_name + '\"').get(obj_property)
        return self.model[obj_class][obj_name].get(obj_property)

    def get_parent(self, obj_name: str, obj_type: str):
        """
        get parent of object
        Parameters
        ----------
        obj_name: str -- name of object
        obj_type: str -- type of object

        Returns
        -------
        parent_name: str
        parent_type: str
        """
        # 1. get name of parent
        parent_name = self.model[obj_type][obj_name].get('parent')
        # 2. get type of parent
        parent_type = None
        if parent_name is not None:
            parent_type = self.get_object_type(parent_name)
        return parent_name, parent_type

    def get_final_parent(self, obj_name: str, obj_type: str):
        """
        get ultimate parent of object
        Parameters
        ----------
        obj_name: str -- name of object
        obj_type: str -- type of object

        Returns
        -------
        parent_name: str
        parent_type: str
        """
        parent_name, parent_type = self.get_parent(obj_name, obj_type)
        if parent_type is not None:
            if self.model[parent_type][parent_name].get('parent') is None:
                return parent_name, parent_type
            else:
                return self.get_final_parent(parent_name, parent_type)

    def model_from_comp(self, comp: set):
        model = {}
        for obj_name in comp:
            if len(obj_name.split(
                    ":")) == 2:  # support receiving obj_name in the form class:obj_name e.g. "meter:node_2"
                obj_class = obj_name.split(":")[0]
                obj_name = obj_name.split(":")[-1]
            else:
                obj_class = self.get_object_type(obj_name)
            # add to model
            if model.get(obj_class) is None:
                model[obj_class] = {}
            if model[obj_class].get(obj_name) is not None:
                warnings.warn(f'Overwriting object, {obj_name}!')
            model[obj_class][obj_name] = self.get(obj_name, obj_class)
            # Find all references to the node:

        return model

    def get_all(self, obj_names: list):
        model = {}
        for obj_name in obj_names:
            if len(obj_name.split(
                    ":")) == 2:  # support receiving obj_name in the form class:obj_name e.g. "meter:node_2"
                obj_class = obj_name.split(":")[0]
                obj_name = obj_name.split(":")[-1]
            else:
                obj_class = self.get_object_type(obj_name)
            # add to model
            if model.get(obj_class) is None:
                model[obj_class] = {}
            if model[obj_class].get(obj_name) is not None:
                warnings.warn(f'Overwriting object, {obj_name}!')
            model[obj_class][obj_name] = self.get(obj_name, obj_class)
        return model

    def get(self, obj_name: str, obj_class=None):
        """
        Parameters
        ----------
        obj_name: str -- object name
        obj_class: str -- optional class of object

        Returns
        -------
        object dictionary
        """
        if len(obj_name.split(":")) == 2:  # support receiving obj_name in the form class:obj_name e.g. "meter:node_2"
            obj_class = obj_name.split(":")[0]
            obj_name = obj_name.split(":")[-1]
        elif obj_class is None:
            obj_class = self.get_object_type(obj_name)
        if self.model[obj_class].get(obj_name) is None:
            return self.model[obj_class].get('\"' + obj_name + '\"')
        return self.model[obj_class][obj_name]

    def run(self, tmp_model_path=None, file_names_to_read=None):
        """
        Run the model in a temporary directory and read the result files into data frames.
        Parameters
        ----------
        tmp_model_path: str or Path -- directory where temporary directory will be created to store model and outputs
        file_names_to_read: list -- list of names of output files to read

        Returns
        -------
        dictionary of results as pandas dataframes
        """

        # 1. Create temporary directory for storing and running the model
        if tmp_model_path is None:
            if self.base_dir_path is None:
                raise RuntimeError("No path is provided to add temporary directory to. Either provide, tmp_model_path "
                                   "or define parameter, base_dir_path")
            tmp_model_path = self.base_dir_path
        tmp_dir = Path(tmp_model_path) / 'gld_tmp'
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)  # remove old temporary directory
        tmp_dir.mkdir()
        output_name = tmp_dir / 'system.glm'
        # 2. Check for players and copy player files
        if self.model.get('player') is not None:
            player_dir = tmp_dir / 'players'
            player_dir.mkdir()
            for player_name in self.model['player'].keys():
                if self.model['player'][player_name].get('file') is not None:
                    old_path_name = (tmp_model_path / Path(self.model['player'][player_name].get('file'))).absolute()
                    shutil.copy(old_path_name, player_dir/old_path_name.name)
                    self.model['player'][player_name]['file'] = str(Path('players/'+old_path_name.name))
        # 3. Create subdirectory for output files to go to.
        out_dir = tmp_dir / 'output'
        out_dir.mkdir()
        self.change_output_dirs('output')

        # 4. Write glm file
        self.write(output_name)
        # 5. Run glm file
        # self.run_gld_on_subprocess(output_name.name, tmp_dir)
        subprocess.run(["gridlabd", output_name.name], env=os.environ, cwd=tmp_dir)
        # 6. Read results
        results = {}
        if file_names_to_read is None:
            for file in list(out_dir.glob('*.csv')):
                results[Path(file).name] = self.read_csv(file)
        else:
            for file in file_names_to_read:
                file = Path('output')/file
                results[Path(file).name] = self.read_csv(file)
        return results

    def total_load(self):
        minus30 = np.exp(-1j*np.pi/6)  # -30 deg
        minus120 = np.exp(-1j*np.pi*2/3)  # -120 deg
        delta_to_wye = minus30/np.sqrt(3) * np.array(
            [
                [1,           0, -1*minus120],
                [-1*minus120, 1,           0],
                [0, -1*minus120,           1]
            ]
        )
        s_wye_tot = np.zeros(3, dtype=complex)
        s_del_tot = np.zeros(3, dtype=complex)
        # parse loads as constant p and q loads on each phase using nominal voltage
        for load_name, load in self.model['load'].items():
            v_nom = float(load.get('nominal_voltage'))
            v = np.zeros(3, dtype=complex)
            v[0] = v_nom * (minus120 ** 0)
            v[1] = v_nom * (minus120 ** 1)
            v[2] = v_nom * (minus120 ** 2)
            v_delta = np.sqrt(3)*v*np.exp(1j*np.pi/6)
            s_wye = np.zeros(3, dtype=complex)
            s_delta = np.zeros(3, dtype=complex)
            # Add all types of loads assuming nominal balanced voltages

            for i, ph in enumerate('ABC'):
                # Constant PQ WYE connected load:
                s_wye[i] += complex(load.get(f'constant_power_{ph}', complex(0)))
                s_wye[i] += complex(load.get(f'constant_power_{ph}N', complex(0)))
                # Constant current WYE connected load:
                s_wye[i] += v[i]*np.conjugate(complex(load.get(f'constant_current_{ph}', complex(0))))
                s_wye[i] += v[i]*np.conjugate(complex(load.get(f'constant_current_{ph}N', complex(0))))
                # Constant impedance WYE connected load
                if f'constant_impedance_{ph}' in load.keys():
                    s_wye[i] += (v[i])**2 / complex(load.get(f'constant_impedance_{ph}', complex(0)))
                if f'constant_impedance_{ph}N' in load.keys():
                    s_wye[i] += (v[i])**2 / complex(load.get(f'constant_impedance_{ph}N', complex(0)))
                # ZIP Loads
                if f'base_power_{ph}' in load.keys():
                    base_power = float(load.get(f'base_power_{ph}'))
                    power_pf = float(load.get(f'power_pf_{ph}', 1))
                    current_pf = float(load.get(f'current_pf_{ph}', 1))
                    impedance_pf = float(load.get(f'impedance_pf_{ph}', 1))
                    power_fraction = float(load.get(f'power_fraction_{ph}', 1))
                    current_fraction = float(load.get(f'current_fraction_{ph}', 0))
                    impedance_fraction = float(load.get(f'impedance_fraction_{ph}', 0))
                    p = base_power * (power_fraction * power_pf +
                                      current_fraction * current_pf +
                                      impedance_fraction * impedance_pf)
                    q = base_power * (power_fraction * np.sin(np.arccos(power_pf)) +
                                      current_fraction * np.sin(np.arccos(current_pf)) +
                                      impedance_fraction * np.sin(np.arccos(impedance_pf)))
                    s_wye[i] += p + 1j*q
            for i, ph in enumerate(['AB', 'BC', 'CA']):
                # Constant PQ delta connected load:
                s_delta[i] = complex(load.get(f'constant_power_{ph}', complex(0)))
                # Constant current WYE connected load:
                s_delta[i] += v_delta[i]*np.conjugate(complex(load.get(f'constant_current_{ph}', complex(0))))
                # Constant impedance WYE connected load
                if f'constant_impedance_{ph}' in load.keys():
                    s_delta[i] += (v_delta[i])**2 / complex(load.get(f'constant_impedance_{ph}', complex(1)))
            # print(f'{load_name}:')
            s_wye_tot += s_wye
            s_del_tot += s_delta
        s_tot = s_wye_tot + delta_to_wye @ s_del_tot
        print(f'Total Load per phase:\n'
              f'{s_tot}')
        print(f'Total Load:\n'
              f'{sum(s_tot)}')
        return s_tot

    def analyze_loads(self):
        simplified_loads = []
        minus30 = np.exp(-1j*np.pi/6)  # -30 deg
        minus120 = np.exp(-1j*np.pi*2/3)  # -120 deg
        delta_to_wye = minus30/np.sqrt(3) * np.array(
            [
                [1,           0, -1*minus120],
                [-1*minus120, 1,           0],
                [0, -1*minus120,           1]
            ]
        )
        s_wye_tot = np.zeros(3, dtype=complex)
        s_del_tot = np.zeros(3, dtype=complex)
        # parse loads as constant p and q loads on each phase using nominal voltage
        for load_name, load in self.model['load'].items():
            v_nom = float(load.get('nominal_voltage'))
            v = np.zeros(3, dtype=complex)
            v[0] = v_nom * (minus120 ** 0)
            v[1] = v_nom * (minus120 ** 1)
            v[2] = v_nom * (minus120 ** 2)
            v_delta = np.sqrt(3)*v*np.exp(1j*np.pi/6)
            s_wye = np.zeros(3, dtype=complex)
            s_delta = np.zeros(3, dtype=complex)
            # Add all types of loads assuming nominal balanced voltages

            if 'D' not in load.get('phases'):
                for i, ph in enumerate('ABC'):
                    # Constant PQ WYE connected load:
                    s_wye[i] += complex(load.get(f'constant_power_{ph}', complex(0)))
                    s_wye[i] += complex(load.get(f'constant_power_{ph}N', complex(0)))
                    # Constant current WYE connected load:
                    s_wye[i] += v[i]*np.conjugate(complex(load.get(f'constant_current_{ph}', complex(0))))
                    s_wye[i] += v[i]*np.conjugate(complex(load.get(f'constant_current_{ph}N', complex(0))))
                    # Constant impedance WYE connected load
                    if f'constant_impedance_{ph}' in load.keys():
                        s_wye[i] += \
                            np.abs(v[i])**2 / np.conjugate(complex(load.get(f'constant_impedance_{ph}', complex(0))))
                    if f'constant_impedance_{ph}N' in load.keys():
                        s_wye[i] += \
                            np.abs(v[i])**2 / np.conjugate(complex(load.get(f'constant_impedance_{ph}N', complex(0))))
                    # ZIP Loads
                    if f'base_power_{ph}' in load.keys():
                        base_power = float(load.get(f'base_power_{ph}'))
                        power_pf = float(load.get(f'power_pf_{ph}', 1))
                        current_pf = float(load.get(f'current_pf_{ph}', 1))
                        impedance_pf = float(load.get(f'impedance_pf_{ph}', 1))
                        power_fraction = float(load.get(f'power_fraction_{ph}', 1))
                        current_fraction = float(load.get(f'current_fraction_{ph}', 0))
                        impedance_fraction = float(load.get(f'impedance_fraction_{ph}', 0))
                        p = base_power * (power_fraction * power_pf +
                                          current_fraction * current_pf +
                                          impedance_fraction * impedance_pf)
                        q = base_power * (power_fraction * np.sin(np.arccos(power_pf)) +
                                          current_fraction * np.sin(np.arccos(current_pf)) +
                                          impedance_fraction * np.sin(np.arccos(impedance_pf)))
                        s_wye[i] += p + 1j*q
            if "D" in load.get('phases'):
                for i, ph in enumerate(['AB', 'BC', 'CA']):
                    # Constant PQ delta connected load:
                    s_delta[i] = complex(load.get(f'constant_power_{ph}', complex(0)))
                    # Constant current WYE connected load:
                    s_delta[i] += v_delta[i]*np.conjugate(complex(load.get(f'constant_current_{ph}', complex(0))))
                    # Constant impedance WYE connected load
                    if f'constant_impedance_{ph}' in load.keys():
                        s_delta[i] += np.abs(v_delta[i]) ** 2 / np.conjugate(
                                complex(load.get(f'constant_impedance_{ph}', complex(1))))

                # repeat since gridlabd reads constant_power_A as constant_power_AB when delta connected
                for i, ph in enumerate(['A', 'B', 'C']):
                    # Constant PQ delta connected load:
                    s_delta[i] = complex(load.get(f'constant_power_{ph}', complex(0)))
                    # Constant current WYE connected load:
                    s_delta[i] += v_delta[i]*np.conjugate(complex(load.get(f'constant_current_{ph}', complex(0))))
                    # Constant impedance WYE connected load
                    if f'constant_impedance_{ph}' in load.keys():
                        s_delta[i] += np.abs(v_delta[i]) ** 2 / np.conjugate(
                                complex(load.get(f'constant_impedance_{ph}', complex(1))))
            print(f'{load_name}:')
            s = s_wye + delta_to_wye @ s_delta
            print(f's: {s}\n'
                  f's_wye: {s_wye},\n'
                  f's_del: {s_delta}')
            s_wye_tot += s_wye
            s_del_tot += s_delta
            name_and_load = [load_name]
            name_and_load.extend(s)
            print(name_and_load)
            simplified_loads.append(name_and_load)
        s_tot = s_wye_tot + delta_to_wye @ s_del_tot
        print(f'Total Load per phase:\n'
              f'{s_tot}')
        print(f'Total Load:\n'
              f'{sum(s_tot)}')
        simplified_loads = pd.DataFrame(simplified_loads)
        return simplified_loads
    # ~~~~~~~~~~ Graphing convenience methods ~~~~~~~~~~~~~~~~~~~

    def analyze(self):
        graph.analyze(self.model)

    def create_graph(self, delete_open=False):
        if delete_open:
            return graph.create_graph(graph.delete_open(self.model))
        else:
            return graph.create_graph(self.model)

    def draw_feeders(self, feeder_swing_nodes: list = None, **options):
        return graph.draw_feeders(self.model, feeder_swing_nodes, **options)

    def draw(self, **options):
        return graph.draw(self.model, **options)
    # ~~~~~~~~~~ Methods for manipulating the model ~~~~~~~~~~~~~
    def rename_object(self, obj_name: str, new_obj_name: str, obj_type: str = None):
        if obj_type is None:
            obj_type = self.get_object_type(obj_name)

        children = self.find_objects_with_property_value('parent', obj_name, prepend_class=True)
        for child in children:
            self.get(child)['parent'] = new_obj_name
        if obj_type in _node_types:
            upstream_links = self.find_objects_with_property_value('to', obj_name, search_types=_link_types,
                                                                   prepend_class=True)
            for upstream_link in upstream_links:
                self.get(upstream_link)['to'] = new_obj_name
            downstream_links = self.find_objects_with_property_value('from', obj_name, search_types=_link_types,
                                                                     prepend_class=True)
            for downstream_link in downstream_links:
                self.get(downstream_link)['from'] = new_obj_name

        configurations = self.find_objects_with_property_value('configuration', obj_name, prepend_class=True)
        for configuration in configurations:
            self.get(configuration)['configuration'] = new_obj_name
        conductor_A_refs = self.find_objects_with_property_value('conductor_A', obj_name,
                                                                 search_types=['line_configuration'],
                                                                 prepend_class=True)
        for ref in conductor_A_refs:
            self.get(ref)['conductor_A'] = new_obj_name
        conductor_B_refs = self.find_objects_with_property_value('conductor_B', obj_name,
                                                                 search_types=['line_configuration'],
                                                                 prepend_class=True)
        for ref in conductor_B_refs:
            self.get(ref)['conductor_B'] = new_obj_name
        conductor_C_refs = self.find_objects_with_property_value('conductor_C', obj_name,
                                                                 search_types=['line_configuration'],
                                                                 prepend_class=True)
        for ref in conductor_C_refs:
            self.get(ref)['conductor_C'] = new_obj_name
        conductor_N_refs = self.find_objects_with_property_value('conductor_N', obj_name,
                                                                 search_types=['line_configuration'],
                                                                 prepend_class=True)
        for ref in conductor_N_refs:
            self.get(ref)['conductor_N'] = new_obj_name
        spacing_refs = self.find_objects_with_property_value('spacing', obj_name, search_types=['line_configuration'],
                                                             prepend_class=True
                                                             )
        for ref in spacing_refs:
            self.get(ref)['spacing'] = new_obj_name

        self.add_object(obj_type, new_obj_name, **self.get(obj_name))
        del self.model[obj_type][obj_name]

    def remove_quotes_from_obj_names(self):
        """
        Use this to remove all quotes from object names and references. They aren't necessary.
        You may want to use this if quotes cause problems for processing.
        """
        model = {}
        for obj_class, class_dict in self.model.items():
            model[obj_class] = {}
            for obj_name, obj_dict in class_dict.items():
                model[obj_class][obj_name.strip('\"').strip('\'')] = obj_dict
        self.model = model
        link_types = [
            'link', 'overhead_line', 'underground_line', 'triplex_line', 'transformer',
            'regulator', 'fuse', 'switch', 'recloser', 'relay', 'sectionalizer', 'series_reactor'
        ]

        # Remove Quotes from references as well
        for link_type in link_types:
            if self.model.get(link_type) is not None:
                for link_name in self.model[link_type].keys():
                    if self.model[link_type][link_name].get('from') is not None:
                        self.model[link_type][link_name]['from'] = \
                            self.model[link_type][link_name]['from'].strip('\"').strip('\'')
                    if self.model[link_type][link_name].get('to') is not None:
                        self.model[link_type][link_name]['to'] = \
                            self.model[link_type][link_name]['to'].strip('\"').strip('\'')
                    # clean configuration references
                    if link_type in ['overhead_line', 'underground_line', 'transformer', 'regulator']:
                        if self.model[link_type][link_name].get('configuration') is not None:
                            self.model[link_type][link_name]['configuration'] = \
                                self.model[link_type][link_name]['configuration'].strip('\"').strip('\'')

        node_types = ['meter', 'node', 'triplex_node', 'triplex_meter', 'load', 'pqload', 'capacitor', 'recorder',
                      'inverter', 'diesel_dg']
        for obj_type in node_types:
            if self.model.get(obj_type) is not None:
                for obj_name in self.model[obj_type].keys():
                    if self.model[obj_type][obj_name].get('parent') is not None:
                        self.model[obj_type][obj_name]['parent'] = \
                            self.model[obj_type][obj_name]['parent'].strip('\"').strip('\'')
        # remove quotes from all line_configuration properties since they are all links to other objects.
        if self.model.get('line_configuration') is not None:
            for obj_name in self.model['line_configuration'].keys():
                for obj_property in self.model['line_configuration'][obj_name].keys():
                    self.model['line_configuration'][obj_name][obj_property] = \
                        self.model['line_configuration'][obj_name][obj_property].strip('\"').strip('\'')

    def change_output_dirs(self, new_output_dir):
        """
        Modify all the output file paths to have the path provided.

        Parameters
        ----------
        new_output_dir: str or Path -- directory to send all output files to.
        """
        # filename in voltdump, currdump, impedance_dump
        # file in recorder, collector, group_recorder
        for o_type in ['voltdump', 'currdump', 'impedance_dump']:
            if self.model.get(o_type) is not None:
                for o_name in self.model[o_type].keys():
                    if self.model[o_type][o_name].get('filename') is not None:
                        original_path = Path(self.model[o_type][o_name].get('filename'))
                        new_path = Path(new_output_dir) / original_path.name
                        self.model[o_type][o_name]['filename'] = new_path
        for o_type in ['recorder', 'collector', 'group_recorder', 'multi_recorder']:
            if self.model.get(o_type) is not None:
                for o_name in self.model[o_type].keys():
                    if self.model[o_type][o_name].get('file') is not None:
                        original_path = Path(self.model[o_type][o_name].get('file'))
                        new_path = Path(new_output_dir) / original_path.name
                        self.model[o_type][o_name]['file'] = new_path

    def change_player_dirs(self, new_player_dir):
        o_type = 'player'
        if self.model.get(o_type) is not None:
            for o_name in self.model[o_type].keys():
                if self.model[o_type][o_name].get('file') is not None:
                    original_path = Path(self.model[o_type][o_name].get('file'))
                    new_path = Path(new_player_dir) / original_path.name
                    self.model[o_type][o_name]['file'] = new_path

    def add_object(self, obj_type, obj_name, **params):
        """
        A convenience function for adding an object to the model. This will overwrite existing objects.

        Parameters
        ----------
        obj_type: str -- type of object
        obj_name: str -- name of object
        params: Keyword arguments become parameters of the object.
                Some property names are not allowed as keywords in Python.
                To get around this problem, pass the parameters as a dictionary with ** in front:
                add_object(obj_type, obj_name, **{'from': 'bus_3', 'to': 'bus_4', ...}).
        """
        if self.model.get(obj_type) is None:
            self.model[obj_type] = {}
        if self.model[obj_type].get(obj_name) is not None:
            warnings.warn(f'Overwriting object, {obj_name}!')
        self.model[obj_type][obj_name] = params

    def add_module(self, module_name, **params):
        """
        A convenience function for adding a module. If the module already exists it will overwrite existing parameters.
        Parameters
        ----------
        module_name: str -- name of module to add
        params: Keyword arguments become parameters of the module.
                Some property names are not allowed as keywords in Python.
                To get around this problem, pass the parameters as a dictionary with ** in front:
                add_module(obj_type, obj_name, **{property1: prop_val1, ...}).
        """
        if self.modules.get(module_name) is None:
            self.modules[module_name] = params
        else:
            warnings.warn(f'Overwriting module, {module_name}, parameters!')
            self.modules[module_name] = params

    def require_module(self, module_name, **params):
        """
        Will ensure that the module is included. If not it will be added. If it is already included it will do nothing.
        This is similar to add_module but does not overwrite parameters if it already exists
        Parameters
        ----------
        module_name
        params

        """
        if self.modules.get(module_name) is None:
            self.modules[module_name] = params

    def add_helics(self, federate_name, config_path):
        """
        Add everything the model needs to enable HELICS use with GridLAB-D
        Parameters
        ----------
        federate_name: str -- name of federate
        config_path: str or Path -- path to HELICS configuration file

        Returns
        -------

        """
        self.require_module('connection')
        self.add_object('helics_msg', federate_name, configure=Path(config_path).as_posix())

    def remove_helics(self):
        """
        Remove HELICS from model so it can run independently.
        """
        if self.model.get('helics_msg') is not None:
            del self.model['helics_msg']

    def rename_all_nodes(self, prefix=None):
        if prefix is None:
            prefix = "n"
        swing_nodes = self.swing_nodes()
        if len(swing_nodes) != 1:
            warnings.warn("This method assumes the model has a single SWING bus.")
            return
        if len(swing_nodes) == 1:
            root = swing_nodes[0]
        g = self.create_graph(delete_open=True)
        g = graph.fix_reversed_links(g, root)
        node_gen = nx.dfs_preorder_nodes(g, source=root)
        for i, n, in enumerate(node_gen):

            new_name = prefix + f"{i+1}"
            self.rename_object(n, new_name)
            print(f"{i+1}: {n}: {new_name}")

    def rename_all_overhead_lines(self, prefix=None):

        if prefix is None:
            prefix = "ohl"
        lines = [line for line in self.model.get('overhead_line').keys()]
        for line in lines:
            n_from = self.model['overhead_line'][line]['from']
            n_to = self.model['overhead_line'][line]['to']
            new_name = f"{prefix}_{n_from}_{n_to}"
            self.rename_object(line, new_name, 'overhead_line')

    def rename_all_fuses(self, prefix=None):

        if prefix is None:
            prefix = "fuse"
        fuses = [fuse for fuse in self.model.get('fuse').keys()]
        for fuse in fuses:
            n_from = self.model['fuse'][fuse]['from']
            n_to = self.model['fuse'][fuse]['to']
            new_name = f"{prefix}_{n_from}_{n_to}"
            self.rename_object(fuse, new_name, 'fuse')

    def rename_all_loads(self, prefix=None):
        if prefix is None:
            prefix = "load"
        loads = [line for line in self.model.get('load').keys()]
        for load in loads:
            parent = self.model['load'][load].get('parent')
            if parent is None:
                warnings.warn("Load is not a child object. This method assumes loads have parent nodes.")
                return
            new_name = f"{prefix}_{parent}"
            self.rename_object(load, new_name, 'load')

    # TODO: add method for combining GLMs. Duplicate could be optionally deleted or renamed.

    # ~~~~~~~~~~ Static Methods ~~~~~~~~~~~~~
    @staticmethod
    def read_csv(filepath, **kwargs):
        """
        Read GridLAB-D output csv file into a dataframe. This will automatically choose the appropriate header line.
        Parameters
        ----------
        filepath

        """
        try:
            df = pd.read_csv(
                filepath,
                sep=',',
                header=1, index_col=0, **kwargs)
        except pd.errors.ParserError:
            df = pd.read_csv(
                filepath,
                sep=',',
                header=8, index_col=0, **kwargs)
        return df
