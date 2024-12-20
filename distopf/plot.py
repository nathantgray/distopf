"""
plot.py - Plotly visualization functions for LinDistModel results
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from distopf import LinDistModel

# plot color codes for distribution system plots
COLORCODES = {
    # key: circuit element type
    # values: color, linewidth
    "transformer": ["black", 4, "solid"],
    "switch": ["black", 4, "solid"],
    "line": ["black", 4, "solid"],
    "reactor": ["black", 4, "solid"],
}


def plot_voltages(v: pd.DataFrame = None) -> go.Figure:
    """
    Parameters
    ----------
    v : pd.DataFrame, Dataframe containing solved voltages for each bus.
        Typically generated by the LinDistModel.get_voltages() method.

    Returns
    -------
    fig : Plotly figure object
        Plotly figure object containing the voltage magnitudes for each bus.
    """
    v = v.melt(ignore_index=False, id_vars="name", var_name="phase", value_name="v")
    fig = px.scatter(v, x=v.name, y="v", facet_col="phase", color="phase")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.for_each_xaxis(lambda a: a.update(title="Bus Name"))
    return fig


def compare_voltages(v1: pd.DataFrame, v2: pd.DataFrame) -> go.Figure:
    """
    Visually compare voltages by plotting two different results.
    Parameters
    ----------
    v1 : pd.DataFrame
    v2 : pd.DataFrame

    Returns
    -------
    fig : Plotly figure object
    """
    v1 = v1.melt(ignore_index=True, var_name="phase", id_vars=["name"], value_name="v1")
    v2 = v2.melt(ignore_index=True, var_name="phase", id_vars=["name"], value_name="v2")
    v = pd.merge(v1, v2, on=["name", "phase"])
    fig = px.line(v, x="name", facet_col="phase", y=["v1", "v2"], markers=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.for_each_xaxis(lambda a: a.update(title="Bus Name"))
    return fig


def voltage_differences(v1: pd.DataFrame, v2: pd.DataFrame) -> go.Figure:
    """
    Visually compare voltages from two different results by plotting the difference v1-v2.
    Parameters
    ----------
    v1 : pd.DataFrame
    v2 : pd.DataFrame

    Returns
    -------
    fig : Plotly figure object
    """
    v1["id"] = v1.index
    v2["id"] = v2.index
    v1 = v1.melt(ignore_index=True, var_name="phase", id_vars=["id"], value_name="v1")
    v2 = v2.melt(ignore_index=True, var_name="phase", id_vars=["id"], value_name="v2")
    v = pd.merge(v1, v2, on=["id", "phase"])
    v["diff"] = v["v1"] - v["v2"]
    fig = px.line(v, x="id", y="diff", facet_col="phase")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.for_each_xaxis(lambda a: a.update(title="Bus Name"))
    return fig


def plot_power_flows(s: pd.DataFrame) -> go.Figure:
    """
    Plot the active and reactive power flowing into each bus on each phase.
    Parameters
    ----------
    s : pd.DataFrame

    Returns
    -------
    fig : Plotly figure object
    """

    s = s.melt(
        ignore_index=True, id_vars=["fb", "tb"], var_name="phase", value_name="s"
    )
    s["p"] = s.s.apply(lambda x: x.real)
    s["q"] = s.s.apply(lambda x: x.imag)
    del s["s"]
    s = s.melt(
        ignore_index=True,
        id_vars=["fb", "tb", "phase"],
        var_name="part",
        value_name="power",
    )
    fig = px.bar(
        s,
        x="tb",
        y="power",
        facet_col="phase",
        facet_row="part",
        color="phase",
        labels={"tb": "To-Bus Name"},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.update_layout(
        yaxis4_title="Active Power (p.u.)",
        yaxis_title="Reactive Power (p.u.)",
    )
    return fig


def compare_flows(s1: pd.DataFrame, s2: pd.DataFrame) -> go.Figure:
    """
    Similar to plot_power_flows but plots two results side by side.
    Parameters
    ----------
    s1 : pd.DataFrame
    s2 : pd.DataFrame

    Returns
    -------
    fig : Plotly figure object
    """
    s1 = s1.melt(
        ignore_index=True, id_vars=["fb", "tb"], var_name="phase", value_name="s"
    )
    s1["p"] = s1.s.apply(lambda x: x.real)
    s1["q"] = s1.s.apply(lambda x: x.imag)
    del s1["s"]
    s1 = s1.melt(
        ignore_index=True,
        id_vars=["fb", "tb", "phase"],
        var_name="part",
        value_name="s1",
    )
    s2 = s2.melt(
        ignore_index=True, id_vars=["fb", "tb"], var_name="phase", value_name="s"
    )
    s2["p"] = s2.s.apply(lambda x: x.real)
    s2["q"] = s2.s.apply(lambda x: x.imag)
    del s2["s"]
    s2 = s2.melt(
        ignore_index=True,
        id_vars=["fb", "tb", "phase"],
        var_name="part",
        value_name="s2",
    )
    s = pd.merge(s1, s2, on=["fb", "tb", "phase", "part"], how="outer")
    fig = px.bar(
        s,
        x="tb",
        y=["s1", "s2"],
        facet_col="phase",
        facet_row="part",
        barmode="group",
        labels={"tb": "To-Bus Name"},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.update_layout(
        yaxis4_title="Active Power (p.u.)",
        yaxis_title="Reactive Power (p.u.)",
    )
    return fig


def plot_ders(ders: pd.DataFrame) -> go.Figure:
    """
    Plot the generated power for each DER.
    Parameters
    ----------
    ders : pd.DataFrame

    Returns
    -------
    fig : Plotly figure object
    """
    dec_var = ders.melt(
        ignore_index=False,
        var_name="phase",
        value_name="Generated Power (p.u.)",
        id_vars="name",
    )
    fig = px.bar(
        dec_var,
        x=dec_var.name,
        y="Generated Power (p.u.)",
        color="phase",
        barmode="group",
        labels={"name": "DER Bus Name"},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]).upper())
    return fig


def plot_network(
    model: LinDistModel,
    v: pd.DataFrame = None,
    s: pd.DataFrame = None,
    control_values: pd.DataFrame = None,
    control_variable: str = "",
    v_min: int = 0.95,
    v_max: int = 1.05,
    show_phases: str = "abc",
    show_reactive_power: bool = False,
) -> go.Figure:
    """
    Plot the distribution network showing voltage and power results.
    Parameters
    ----------
    model : LinDistModel
    v : pd.DataFrame, (default=None) Dataframe containing voltage magnitudes for each bus.
    s : pd.DataFrame, (default=None) Dataframe containing power flows for each branch.
    v_min : (default=0.95) Used for scaling node colors.
    v_max : (default=1.05) Used for scaling node colors.
    show_phases : (default="abc") valid options: "a", "b", "c", or "abc"
    show_reactive_power : (default=False) If True, show reactive power flows instead of active power flows.

    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    _s = s.copy()
    _v = v.copy()
    # validate phases
    if show_phases.lower() not in ["a", "b", "c", "abc"]:
        raise ValueError("Invalid phase. Must be 'a', 'b', 'c', or 'abc'.")
    show_phases = show_phases.lower()
    phase_list = sorted([ph.lower() for ph in show_phases])
    bus_data = model.bus.copy()
    branch_data = model.branch.copy()
    gen_data = model.gen.copy()
    cap_data = model.cap.copy()
    node_size = 10
    edge_scale = 10
    edge_min = 1
    bus_data["y"] = bus_data.latitude - bus_data.latitude.mean()
    bus_data["x"] = bus_data.longitude - bus_data.longitude.mean()
    if bus_data.x.abs().max() > 0:
        bus_data.x = bus_data.x / bus_data.x.abs().max()
    if bus_data.y.abs().max() > 0:
        bus_data.y = bus_data.y / bus_data.y.abs().max()
    branch_data = branch_data.loc[branch_data.status != "OPEN", :]
    node_trace = go.Scatter(
        x=bus_data["x"],
        y=bus_data["y"],
        mode="markers",
        marker=dict(
            showscale=_v is not None,
            cmin=v_min,
            cmax=v_max,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' | 'Aggrnyl'
            colorscale="Viridis",
            reversescale=False,
            size=node_size,
            color="white",
            line_width=node_size / 5,
            line_color="black",
        ),
        showlegend=False,
        text=bus_data["name"],
        hovertemplate="%{text}",
        textposition="top center",
    )
    # Add substation marker
    substation_buses = bus_data.loc[bus_data.bus_type == "SWING", :]
    substation_trace = go.Scatter(
        x=substation_buses["x"],
        y=substation_buses["y"],
        mode="markers",
        marker=dict(
            symbol="square",
            size=node_size * 2,
            color="black",
            line_width=1,
            line_color="black",
        ),
        showlegend=True,
        name="Substation",
        hoverinfo="none",
    )
    # Add generator markers
    gen_buses = bus_data.loc[bus_data.id.isin(gen_data.id), :]
    gen_trace = go.Scatter(
        x=gen_buses["x"],
        y=gen_buses["y"],
        mode="markers",
        marker=dict(
            symbol="square",
            size=node_size * 2,
            color="white",
            line_width=1,
            line_color="black",
        ),
        showlegend=True,
        name="Generators",
        hoverinfo="none",
    )
    # Add capacitor markers
    cap_buses = bus_data.loc[bus_data.id.isin(cap_data.id), :]
    cap_trace = go.Scatter(
        x=cap_buses["x"],
        y=cap_buses["y"],
        mode="markers",
        marker=dict(
            symbol="star-diamond",
            size=node_size * 2,
            color="white",
            line_width=1,
            line_color="black",
        ),
        showlegend=True,
        name="Capacitors",
        hoverinfo="none",
    )

    text = [f"Bus: '{name}':" for name in bus_data["name"]]
    if _v is not None:
        node_trace.marker.color = _v[phase_list].mean(axis=1)
        for i, bus_row in enumerate(bus_data.itertuples()):
            va = _v.loc[bus_row.id, "a"]
            vb = _v.loc[bus_row.id, "b"]
            vc = _v.loc[bus_row.id, "c"]
            text[i] = text[i] + f"<br>\t|V|:    {va:.3f}  {vb:.3f}  {vc:.3f}"
    for i, bus_row in enumerate(bus_data.itertuples()):
        pla = bus_row.pl_a
        plb = bus_row.pl_b
        plc = bus_row.pl_c
        qla = bus_row.ql_a
        qlb = bus_row.ql_b
        qlc = bus_row.ql_c
        text[i] += f"<br>P-Load: {pla:.3f}  {plb:.3f}  {plc:.3f}"
        text[i] += f"<br>Q-Load: {qla:.3f}  {qlb:.3f}  {qlc:.3f}"

    if cap_data is not None:
        for i, bus_row in enumerate(bus_data.itertuples()):
            if bus_row.id in cap_data.id.to_numpy():
                q_cap = cap_data.loc[
                    cap_data.id == bus_row.id, ["qa", "qb", "qc"]
                ].to_numpy()[0]
                text[
                    i
                ] += f"<br>Cap Q:    {q_cap[0]:.3f}  {q_cap[1]:.3f}  {q_cap[2]:.3f}"

    for i, bus_row in enumerate(bus_data.itertuples()):
        if bus_row.id in gen_data.id.to_numpy():
            p_gen = gen_data.loc[
                gen_data.id == bus_row.id, ["pa", "pb", "pc"]
            ].to_numpy()[0]
            q_gen = gen_data.loc[
                gen_data.id == bus_row.id, ["qa", "qb", "qc"]
            ].to_numpy()[0]
            if control_variable.lower() == "q" and control_values is not None:
                q_gen = control_values.loc[
                    control_values.name == bus_row.name, ["a", "b", "c"]
                ].to_numpy()[0]
            if control_variable.lower() == "p" and control_values is not None:
                p_gen = control_values.loc[
                    control_values.name == bus_row.name, ["a", "b", "c"]
                ].to_numpy()[0]
            text[i] += f"<br>Gen P:    {p_gen[0]:.3f}  {p_gen[1]:.3f}  {p_gen[2]:.3f}"
            text[i] += f"<br>Gen Q:    {q_gen[0]:.3f}  {q_gen[1]:.3f}  {q_gen[2]:.3f}"
    # Create lines for edges
    if _s is not None:
        _s["p_abs"] = np.abs(np.real(_s.loc[:, phase_list].sum(axis=1)))
        _s["q_abs"] = np.abs(np.imag(_s.loc[:, phase_list].sum(axis=1)))
        _s["p_norm"] = (
            _s["p_abs"].to_numpy() / _s["p_abs"].max() * edge_scale + edge_min
        )
        _s["q_norm"] = (
            _s["q_abs"].to_numpy() / _s["q_abs"].max() * edge_scale + edge_min
        )
        _s["p_direction"] = np.sign(np.real(_s.loc[:, phase_list].sum(axis=1)) + 1e-6)
        _s["q_direction"] = np.sign(np.imag(_s.loc[:, phase_list].sum(axis=1)) + 1e-6)
        for i, bus_row in enumerate(bus_data.itertuples()):
            # tb_name = bus_row.name
            tb = bus_row.name
            if tb not in _s.tb.to_numpy():
                continue
            s_edge = _s.loc[_s.tb == bus_row.name].to_dict(orient="list")
            fb = s_edge["fb"][0]
            # fb_name = bus_data.loc[bus_data.id == fb, "name"].values[0]
            sa, sb, sc = s_edge["a"][0], s_edge["b"][0], s_edge["c"][0]
            new_text = (
                f"<br> Branch {fb}→{tb}"
                f"<br>    P flow:  {np.real(sa):.3f}  {np.real(sb):.3f}  {np.real(sc):.3f}"
                f"<br>    Q flow:  {np.imag(sa):.3f}  {np.imag(sb):.3f}  {np.imag(sc):.3f}"
            )
            text[i] = text[i] + new_text
            node_trace.text = text
    edge_traces = []
    arrow_list = []
    reverse_list_x = []
    reverse_list_y = []

    for _, edge in branch_data.iterrows():
        source_data = bus_data[bus_data["id"] == edge["fb"]]
        target_data = bus_data[bus_data["id"] == edge["tb"]]
        x0, x1 = source_data["x"].values[0], target_data["x"].values[0]
        y0, y1 = source_data["y"].values[0], target_data["y"].values[0]
        linewidth = COLORCODES[edge["type"]][1]
        dash = COLORCODES[edge["type"]][2]
        direction = 1
        if _s is not None:
            s_edge = _s.loc[edge.tb, :]
            linewidth = s_edge.p_norm
            direction = s_edge.p_direction
            if show_reactive_power:
                linewidth = s_edge.q_norm
                direction = s_edge.q_direction
            if (
                show_phases.lower() != "abc"
                and show_phases.lower() not in edge.phases.lower()
            ):
                dash = "dot"
        # dash="solid"
        color = "darkblue"
        if direction < 0:
            color = "maroon"
            reverse_list_x.append(1 / 2 * (x0 + x1))
            reverse_list_y.append(1 / 2 * (y0 + y1))
        edge_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=linewidth, dash=dash),
            showlegend=False,
        )
        # if dash != "dot":

        # color = "gray"
        # arrow = _create_arrow(
        #     x0, y0, x1, y1, color, linewidth, direction, node_size
        # )
        # arrow_list.append(arrow)
        edge_traces.append(edge_trace)
    reverse_trace = go.Scatter()
    if not show_reactive_power:
        reverse_trace = go.Scatter(
            x=reverse_list_x,
            y=reverse_list_y,
            mode="markers",
            marker=dict(
                symbol="star-triangle-up",
                size=node_size * 1,
                color="orange",
                line_width=0.5,
                line_color="black",
            ),
            showlegend=True,
            name="Reverse Power Flow",
            hoverinfo="none",
        )
    node_trace.text = text
    fig = go.Figure(
        data=[
            *edge_traces,
            substation_trace,
            reverse_trace,
            cap_trace,
            gen_trace,
            node_trace,
        ]
    )

    title = f"<b>Network Plot (P.U.)</b>"
    if _v is not None:
        title = title + "<br>Node color: "
        if show_phases == "abc":
            title = title + "Average voltage magnitude"
        else:
            title = title + f"Phase-{show_phases.upper()} voltage magnitude"
    if _s is not None:
        title = title + "<br>Line width:"
        if show_phases == "abc":
            title = title + " Total"
        else:
            title = title + f" Phase-{show_phases.upper()}"
        if show_reactive_power:
            title = title + " <i>reactive</i> power flow"
        else:
            title = title + " <i>active</i> power flow"
        title = title + " (reverse flow in red)."
    for arrow in arrow_list:
        fig.add_annotation(arrow)
    fig.update_layout(
        title=title,
        plot_bgcolor="White",
        paper_bgcolor="White",
        title_font_color="Black",
        legend_font_color="Black",
        legend_bgcolor="White",
        margin=dict(t=50, b=10, l=10, r=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(x=0.8, y=0.9),
        font=dict(family="Droid Sans Mono", color="Black"),
        # annotations=arrow_list[0:1],
    )

    return fig


def _create_arrow(x0, y0, x1, y1, color, linewidth, direction, node_size):
    if direction < 0:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    # x1 = x1 - 0.1 * (x1 - x0) / 2
    # y1 = y1 - 0.1 * (y1 - y0) / 2
    x0_new = x0 + 0.0 * (x1 - x0)
    y0_new = y0 + 0.0 * (y1 - y0)
    x1_new = x1 - 0.0 * (x1 - x0)
    y1_new = y1 - 0.0 * (y1 - y0)
    arrow = go.layout.Annotation(
        x=x1_new,
        y=y1_new,
        ax=x0_new,
        ay=y0_new,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=4,
        arrowsize=2 / 3,
        arrowwidth=linewidth,
        arrowcolor=color,
        standoff=node_size / 2,
    )
    return arrow
