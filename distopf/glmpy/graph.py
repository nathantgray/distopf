import warnings
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def create_graph(model):
    """
    Create a networkx graph from the GridLAB-D model.
    Parameters
    ----------
    model: dict

    Returns
    -------
    g: networkx.MultiDiGraph
    """
    g = nx.MultiDiGraph()
    swing = []
    # Add nodes
    node_objects = ['meter', 'node', 'triplex_node', 'triplex_meter', 'load', 'pqload', 'capacitor']
    for obj_type in node_objects:
        if model.get(obj_type) is not None:
            for key, value in model[obj_type].items():
                if model[obj_type][key].get('parent') is None:
                    g.add_node(key, obj_type=obj_type, attributes=value)
                if model[obj_type][key].get('bustype') == 'SWING':
                    swing.append(key)

    # Prepare and add edges to graph
    link_objects = [
        'link', 'overhead_line', 'underground_line', 'triplex_line', 'transformer',
        'regulator', 'fuse', 'switch', 'recloser', 'relay', 'sectionalizer', 'series_reactor'
    ]

    for link_object in link_objects:
        if model.get(link_object) is not None:
            for key, value in model[link_object].items():
                color_a = 0.7
                color_b = 0.7
                color_c = 0.7
                if 'a' in value['phases'].lower():
                    color_a = 0.
                if 'b' in value['phases'].lower():
                    color_b = 0.
                if 'c' in value['phases'].lower():
                    color_c = 0.
                if link_object in ['switch', 'recloser', 'relay', 'sectionalizer']:
                    color_a = 0.95
                    color_b = 0.95
                    color_c = 0.95
                    if 'a' in value['phases'].lower():
                        color_a = 0.5
                    if 'b' in value['phases'].lower():
                        color_b = 0.5
                    if 'c' in value['phases'].lower():
                        color_c = 0.5
                g.add_edge(value['from'], value['to'], obj_type=link_object, name=key, attributes=value, color=(color_a, color_b, color_c))
    return g


def reverse_edge(edge):
    """
    reverses direction of edge
    Parameters
    ----------
    edge: tuple

    Returns
    -------
    reversed edge: tuple
    """
    assert len(edge) >= 2
    if len(edge) == 2:
        return (edge[1], edge[0])
    if len(edge) > 2:
        return (edge[1], edge[0], *edge[2:])


def fix_reversed_links(g, source):
    """
    Attempts to force all edges to point away from source.
    Parameters
    ----------
    g: graph
    source: str -- name of swing node.

    Returns
    -------
    digraph: nx.MultiDiGraph
    """
    digraph = nx.MultiDiGraph()
    if source in g:
        fixed_edges = [edge for edge in nx.dfs_edges(nx.Graph(g), source=source)]
        edge_attr = [g.get_edge_data(edge[0], edge[1]) or g.get_edge_data(edge[1], edge[0]) for edge in fixed_edges]
        edge_dicts = [attr[0] for attr in edge_attr]
        edge_list = []
        for edge, dct in zip(fixed_edges, edge_dicts):
            edge_list.append((edge[0], edge[1], dct))
        digraph.add_edges_from(edge_list)
        # print(len(digraph.nodes))
        missing_edges = []
        for edge in g.edges:
            if edge not in digraph.edges and reverse_edge(edge) not in digraph.edges:
                missing_edges.append((edge[0], edge[1], g.get_edge_data(edge[0], edge[1])))
        add_edges = []
        for edge in missing_edges:
            if edge[0] == source:
                d0 = 0
            else:
                d0 = nx.resistance_distance(nx.Graph(digraph), edge[0], source)
            if edge[1] == source:
                d1 = 0
            else:
                d1 = nx.resistance_distance(nx.Graph(digraph), edge[1], source)
            if d0 < d1:
                add_edges.append((edge[0], edge[1], g.get_edge_data(edge[0], edge[1])[0]))
            else:
                add_edges.append(reverse_edge((edge[0], edge[1], g.get_edge_data(edge[0], edge[1])[0])))
        digraph.add_edges_from(add_edges)
        # data = [G.get_edge_data(edge[0], edge[1]) or G.get_edge_data(edge[1], edge[0])  for edge in digraph.edges]
        return digraph
    warnings.warn("Warning: no source found in this graph!")
    return g


def analyze(model):
    """
    Prints an analysis of the model in the terminal
    Parameters
    ----------
    model: dict

    Returns
    -------
    None
    """
    model = model.copy()
    # Count Objects
    count = {}
    print("Number of objects in entire model:")
    print("----------------------------------")
    for key in model.keys():
        count[key] = len(model[key])
        print(key, ':', len(model[key]))
    print("Swing buses in model:")
    print("----------------------------------")
    # Count Swing Buses
    swing_buses = []
    for obj_type in ['meter', 'node']:
        if obj_type in model.keys():
            for mtrid in model.get(obj_type):
                if model[obj_type][mtrid].get('bustype') == 'SWING':
                    print(f'Swing bus: ', mtrid)
                    swing_buses.append(mtrid)
    print("----------------------------------")
    print('Analyzing isolated compoonents of the system graph:')
    print('')
    print("----------------------------------")
    print('removing open switches ...')
    model_ = delete_open(model)
    g = create_graph(model_)
    print('Weakly connected components:', nx.number_weakly_connected_components(g))
    comp = nx.weakly_connected_components(g)
    comp_list = []
    # loops = []
    for set_ in comp:
        g_ = g.subgraph(set_)
        comp_list.append(g_)
        print(f'Component with {len(g_.nodes())} buses:')
        for swing_bus in swing_buses:
            if swing_bus in g_.nodes():
                print(f"Swing bus, {swing_bus} is in this component")
    print("----------------------------------")


def delete_open(model, verbose=False):
    """
    Returns a copy of the model with open switches, reclosers, and fuses removed.

    Parameters
    ----------
    model: dict

    Returns
    -------
    Returns a copy of the model with open switches, reclosers, and fuses removed.
    """
    model = model.copy()
    for obj_type in ['switch', 'recloser', 'fuse']:
        if obj_type in model.keys():
            switch_model = model[obj_type].copy()
            n_switches = len(model[obj_type])
            for sw in model[obj_type]:
                if model[obj_type][sw].get('status') == 'OPEN':
                    del switch_model[sw]
            model[obj_type] = switch_model.copy()
            n_closed_sw = len(model[obj_type])
            n_open_sw = n_switches - n_closed_sw
            if verbose:
                print(f'{n_switches} open and closed {obj_type} objects')
            print(f'{n_closed_sw} closed {obj_type} objects')
            print(f'{n_open_sw} open {obj_type} objects')

    if 'sectionalizer' in model.keys():
        sectionalizer_model = model['sectionalizer'].copy()
        n_sec = len(model["switch"])
        print(f'{n_sec} open and closed sectionalizers')
        for sec in model['sectionalizer']:
            for key in model['sectionalizer'][sec]:
                if 'state' in key:  # This could break if a secionalizer has open and closed phases
                    if model['sectionalizer'][sec][key] == 'OPEN':
                        del sectionalizer_model[sec]
                        break
        model['sectionalizer'] = sectionalizer_model.copy()
        n_closed_sec = len(model["sectionalizer"])
        n_open_sec = n_sec - n_closed_sec
        print(f'{n_closed_sec} closed sectionalizers')
        print(f'{n_open_sec} open sectionalizers')

    return model


def draw_model(model, **options):
    """
    Plots the model without modifying it.
    Plots do not include open switches.
    Parameters
    ----------
    model: dict -- Gridlabd.model
    options: optional keywords for networkx.draw_networkx

    Returns
    ----------
    None
    """
    # print(f'options = {options}')
    model_ = delete_open(model)
    g = create_graph(model_)
    draw_graph(g, **options)

def draw_graph(g, **options):
    """
    Plots the model without modifying it.
    Plots do not include open switches.
    Parameters
    ----------
    g: graph
    options: optional keywords for networkx.draw_networkx

    Returns
    -------
    None
    """
    edge_color = [g.get_edge_data(edge[0], edge[1])[0]['color'] for edge in g.edges]

    # set default graphing options
    options['with_labels'] = options.get('with_labels', True)
    options['font_size'] = options.get('font_size', 5)
    options['font_color'] = options.get('font_color', 'r')
    options['arrows'] = options.get('arrows', True)
    options['edge_color'] = options.get('edge_color', edge_color)
    options['node_color'] = options.get('node_color', 'black')
    options['node_size'] = options.get('node_size', 1.5)
    options['width'] = options.get('width', 1.5)
    options['arrowstyle'] = options.get('arrowstyle', '-|>')
    options['arrowsize'] = options.get('arrowsize', 6)

    # fig, ax = plt.subplots(1, 1, figsize=(16, 7), dpi=120)
    pos = graphviz_layout(g, prog='dot')
    nx.draw_networkx(g, pos, **options)
    plt.show()


def draw_feeders(model: dict, feeder_swing_nodes: list = None, **options):
    """
    Plots each weakly connected component of the graph separately.
    Plots do not include open switches.
    Attempts to make sure all edges are directed away from the swing node.
    Parameters
    ----------
    model: dict -- Gridlabd.model
    feeder_swing_nodes: list -- list of swing nodes
    options: optional keywords for networkx.draw_networkx

    Returns
    -------

    """
    g = create_graph(delete_open(model))
    comp = nx.weakly_connected_components(g)
    if feeder_swing_nodes is None:
        feeder_swing_nodes = []
    for set_ in comp:
        g_raw = g.subgraph(set_)
        root = list(g_raw.nodes)[0]
        for i, source in enumerate(feeder_swing_nodes):
            print('source: ' + str(source))
            if source in set_:
                g_raw = g.subgraph(set_)
                root = source
        g_ = fix_reversed_links(g_raw, root)
        edge_color = [g_.get_edge_data(edge[0], edge[1])[0]['color'] for edge in g_.edges]

        # set default graphing options
        options['with_labels'] = options.get('with_labels', True)
        options['font_size'] = options.get('font_size', 5)
        options['font_color'] = options.get('font_color', 'r')
        options['arrows'] = options.get('arrows', True)
        options['edge_color'] = options.get('edge_color', edge_color)
        options['node_color'] = options.get('node_color', 'black')
        options['node_size'] = options.get('node_size', 1.5)
        options['width'] = options.get('width', 1.5)
        options['arrowstyle'] = options.get('arrowstyle', '-|>')
        options['arrowsize'] = options.get('arrowsize', 6)

        fig, ax = plt.subplots(1, 1)
        pos = graphviz_layout(g_, prog='dot')
        nx.draw_networkx(g_, pos, ax=ax, **options)
        fig.show()


if __name__ == "__main__":
    G = nx.MultiDiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    G.add_edge(1, 5)
    G.add_edge(5, 9)
    G.add_edge(6, 5)
    G.add_edge(6, 7)
    G.add_edge(7, 8)
# reverse(G, [1])
