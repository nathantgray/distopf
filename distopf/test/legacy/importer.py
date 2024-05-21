import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path, PosixPath
import csv


# from decorator import decorator
# from line_profiler import LineProfiler


# @decorator
# def profile_each_line(func, *args, **kwargs):
#     profiler = LineProfiler()
#     profiled_func = profiler(func)
#     try:
#         profiled_func(*args, **kwargs)
#     finally:
#         profiler.print_stats()


def get_delimiter_and_header(file_path, n_bytes=4000):
    with open(file_path, "r") as csvfile:
        data = csvfile.read(n_bytes)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(data).delimiter

        has_header = sniffer.has_header(data)
        csvfile.seek(0)
        header = None
        if has_header:
            header = 0
        n_col = len(data.split("\n")[0].split(delimiter))
        return delimiter, header, n_col


def dir_importer(dir_path, name, col_names):
    dir_path = Path(dir_path)
    data_path = dir_path / f"{name}.csv"
    if not data_path.exists():
        data_path = dir_path / f"{name}.txt"
    if not data_path.exists():
        data_path = dir_path / f"{name}.tsv"
    return file_importer(data_path, col_names)


def file_importer(data_path, col_names):
    assert Path(data_path).exists()
    delimiter, header, n_col = get_delimiter_and_header(file_path=data_path)
    names = None
    if header is None:
        names = col_names[:n_col]
    return old_2_new_style(
        pd.read_csv(
            data_path, sep=delimiter, header=header, names=names, index_col=False
        )
    )


def get_powerdata(data_file):
    col_names = [
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
    if isinstance(data_file, pd.DataFrame):
        return data_file
    if isinstance(data_file, (str, PosixPath)):
        data_file = Path(data_file)  # make sure it is a PosixPath
        assert data_file.exists()
        df = None
        if Path(data_file).is_file():
            df = file_importer(data_file, col_names)
        if Path(data_file).is_dir():
            df = dir_importer(data_file, "powerdata", col_names)
        if df is not None:
            if "has_load" not in df.keys():
                df["has_load"] = (
                    df.loc[:, ["Pa", "Qa", "Pb", "Qb", "Pc", "Qc"]].any(axis=1) != 0
                )
            if "has_gen" not in df.keys():
                df["has_gen"] = df.loc[:, ["PgA", "PgB", "PgC"]].any(axis=1) != 0
            if "has_cap" not in df.keys():
                df["has_cap"] = df.loc[:, ["CapA", "CapB", "CapC"]].any(axis=1) != 0
        return df


def get_branchdata(data_file):
    col_names = [
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
    if isinstance(data_file, pd.DataFrame):
        return data_file
    if isinstance(data_file, (str, PosixPath)):
        data_file = Path(data_file)  # make sure it is a PosixPath
        if not data_file.exists():
            raise FileExistsError(f"{data_file}")
        # assert data_file.exists()
        if Path(data_file).is_file():
            return file_importer(data_file, col_names)
        if Path(data_file).is_dir():
            return dir_importer(data_file, "branchdata", col_names)


def old_2_new_style(df: pd.DataFrame):
    if "id" in df.keys() and "s_base" not in df.keys():
        # df is old style powerdata with power listed in kW. Convert to pu with base MW
        df.iloc[:, 1:13] /= 1000
        df["s_base"] = 1e6
    if "fb" in df.keys() and "z_base" not in df.keys() and "v_ln_base" not in df.keys():
        # df is old style branchdata. Convert to pu
        df.iloc[:, 2:14] /= (4.16 / np.sqrt(3)) ** 2
        df["z_base"] = (4.16 / np.sqrt(3)) ** 2
        df["v_ln_base"] = 4160 / np.sqrt(3)
    return df


def delete_open_switches(branch_data: pd.DataFrame):
    branch_data_ = branch_data.copy()
    if "status" in branch_data_.keys():
        for row_i, row in branch_data_.iterrows():
            if row["status"] == "OPEN":
                branch_data_.drop(row_i, inplace=True)
                print(f"deleted open link:{row['fb']}-{row['tb']}")
    branch_data_.index = range(branch_data_.shape[0])
    return branch_data_


def delete_unconnected_nodes(branch_data: pd.DataFrame, power_data: pd.DataFrame):
    power_data_ = power_data.copy()
    for i, node_id in enumerate(power_data_["id"]):
        if (
            node_id not in branch_data.fb.values
            and node_id not in branch_data.tb.values
        ):
            power_data_.drop(i, inplace=True)
            print(f"deleted lone node: {node_id}")
    power_data_.index = range(power_data_.shape[0])
    return power_data_


def renumber_nodes(branch_data: pd.DataFrame, power_data: pd.DataFrame):
    if ("bus_type" not in power_data.keys()) or power_data.bus_type.isna().any():
        return branch_data, power_data
    power_data_ = power_data.copy()
    branch_data_ = branch_data.copy()
    # 1. Make tree
    g = nx.DiGraph()
    g.add_nodes_from(power_data_["id"])
    for row_i, row in branch_data_.iterrows():
        g.add_edge(row["fb"], row["tb"])
    # 2. node_list = nx.dfs_preorder_nodes(g, swing)
    swing_bus = power_data_[power_data_.bus_type == "SWING"].id.values[0]
    node_list = nx.dfs_preorder_nodes(nx.Graph(g), swing_bus)
    # node_new2old = {}
    node_old2new = {}

    for i, node_name in enumerate(node_list):
        # node_new2old[i + 1] = int(node_name)
        node_old2new[int(node_name)] = i + 1
    for i_row, row in branch_data_.iterrows():
        branch_data_.loc[i_row, "fb"] = node_old2new[int(branch_data_.loc[i_row, "fb"])]
        branch_data_.loc[i_row, "tb"] = node_old2new[int(branch_data_.loc[i_row, "tb"])]
    for i_row, row in power_data_.iterrows():
        power_data_.loc[i_row, "id"] = node_old2new[int(power_data_.loc[i_row, "id"])]

    return branch_data_, power_data_


# @profile_each_line
def combiner(models: list):
    """
    :param models: list of tuples of dataframes -- (branchdata, powerdata)
    :return:
    new_branchdata,
    new_powerdata
    """

    swings = []
    # 1. make graph based on original names
    g = nx.DiGraph()
    for model in models:
        branchdata = model[0]
        powerdata = model[1]
        swings.append(powerdata[powerdata.bus_type == "SWING"]["name"][0])
        for row_i, row in powerdata.iterrows():
            g.add_node(row["name"], **row)
        for row_i, row in branchdata.iterrows():
            fb = powerdata[powerdata["id"] == row["fb"]]["name"].values[0]
            tb = powerdata[powerdata["id"] == row["tb"]]["name"].values[0]
            g.add_edge(fb, tb, **row)
    # assume only one swing bus has a degree of 1 and all others are greater.
    remove_swings = []
    for swing in swings:
        if nx.degree(g, g.nodes)[swing] > 1:
            g.nodes[swings[1]]["bus_type"] = "PQ"
            remove_swings.append(swing)
    for swing in remove_swings:  # remove swings that are now PQ
        swings.remove(swing)
    print(len(swings))
    assert len(swings) == 1
    swing_bus = swings[0]
    # renumber_nodes
    node_list = nx.dfs_preorder_nodes(nx.Graph(g), swings[0])
    new_powerdata = pd.DataFrame()
    new_branchdata = pd.DataFrame()

    for i, node_name in enumerate(node_list):
        g.nodes[node_name]["id"] = i + 1  # give new id
        new_powerdata = pd.concat(
            [new_powerdata, pd.DataFrame(g.nodes[node_name], index=[i])]
        )  # add to dataframe
    for j, edge in enumerate(g.edges):
        # Get new node ids
        fb = new_powerdata[new_powerdata["name"] == edge[0]]["id"]
        tb = new_powerdata[new_powerdata["name"] == edge[1]]["id"]
        # Reassign fb and tb with new ids
        g.edges[edge]["fb"] = fb
        g.edges[edge]["tb"] = tb
        new_branchdata = pd.concat(
            [new_branchdata, pd.DataFrame(g.edges[edge], index=[j])]
        )
    new_branchdata = new_branchdata.sort_values(by="tb")

    return new_branchdata, new_powerdata


def test_combiner():
    trans_power_path = Path(
        "/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/converted/trans/powerdata.csv"
    )
    trans_branch_path = Path(
        "/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/converted/trans/branchdata.csv"
    )
    RIV212_power_path = Path(
        "/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/converted/RIV215/powerdata.csv"
    )
    RIV212_branch_path = Path(
        "/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/converted/RIV215/branchdata.csv"
    )

    trans_power = get_powerdata(trans_power_path)
    trans_branch = get_branchdata(trans_branch_path)
    RIV212_power = get_powerdata(RIV212_power_path)
    RIV212_branch = get_branchdata(RIV212_branch_path)

    branchdata, powerdata = combiner(
        [(trans_branch, trans_power), (RIV212_branch, RIV212_power)]
    )
    branchdata.to_csv("")
    print(branchdata.head())
    print(powerdata.head())


def test_importer():
    # dir_path_old = Path("/home/nathangray/PycharmProjects/CPMACosim_T2/gridlabd/10_ders/")
    in_path = Path(
        "/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/converted/epb_tnd_for_conversion"
    )
    out_path = Path(
        "/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/renumbered/epb_tnd_for_conversion"
    )
    dir_path_old = Path(
        "/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/converted/epb_tnd_for_conversion"
    )
    powerdata = get_powerdata(dir_path_old / "powerdata.csv")
    branchdata = get_branchdata(dir_path_old / "branchdata.csv")
    pd.testing.assert_frame_equal(
        get_branchdata(dir_path_old / "branchdata.csv"), get_branchdata(dir_path_old)
    )
    pd.testing.assert_frame_equal(
        get_powerdata(dir_path_old / "powerdata.csv"), get_powerdata(dir_path_old)
    )
    print(powerdata.shape)
    print(branchdata.shape)
    branchdata = delete_open_switches(branchdata)
    print(branchdata.shape)
    powerdata = delete_unconnected_nodes(branchdata, powerdata)
    branchdata, powerdata = renumber_nodes(branchdata, powerdata)
    branchdata.to_csv(out_path / "branchdata.csv", index=False)
    powerdata.to_csv(out_path / "powerdata.csv", index=False)
    g = nx.DiGraph()
    g.add_nodes_from(powerdata["id"])
    for row_i, row in branchdata.iterrows():
        g.add_edge(row["fb"], row["tb"])
    comp = list(nx.weakly_connected_components(g))
    print(len(comp))


def test_old_style():
    # dir_path_old = Path("/home/nathangray/PycharmProjects/CPMACosim_T2/gridlabd/10_ders/")
    in_path = Path("/home/nathangray/PycharmProjects/CPMACosim_T2/gridlabd/10_ders/")
    # in_path = Path("/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/converted/epb_tnd_for_conversion")
    # out_path = Path("/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/renumbered/epb_tnd_for_conversion")
    # dir_path_old = Path("/home/nathangray/PycharmProjects/glmpy_dev/glmpy/unittest/epb_test/gridlabd/epb_feeders/converted/epb_tnd_for_conversion")
    assert (in_path / "powerdata.txt").exists()
    assert (in_path / "branchdata.txt").exists()
    powerdata = get_powerdata(in_path / "powerdata.txt")
    branchdata = get_branchdata(in_path / "branchdata.txt")
    pass
    # pd.testing.assert_frame_equal(get_branchdata(in_path/'branchdata.csv'), get_branchdata(in_path))
    # pd.testing.assert_frame_equal(get_powerdata(in_path/'powerdata.csv'), get_powerdata(in_path))
    # print(powerdata.shape)
    # print(branchdata.shape)
    # branchdata = delete_open_switches(branchdata)
    # print(branchdata.shape)
    # powerdata = delete_unconnected_nodes(branchdata, powerdata)
    # branchdata, powerdata = renumber_nodes(branchdata, powerdata)
    # branchdata.to_csv(out_path/'branchdata.csv', index=False)
    # powerdata.to_csv(out_path/'powerdata.csv', index=False)
    # g = nx.DiGraph()
    # g.add_nodes_from(powerdata['id'])
    # for row_i, row in branchdata.iterrows():
    #     g.add_edge(row['fb'], row['tb'])
    # comp = list(nx.weakly_connected_components(g))
    # print(len(comp))


if __name__ == "__main__":
    # test_combiner()
    test_old_style()
