import numpy as np
import pandas as pd
import plotly.express as px

s_df = pd.read_csv("s.csv")
v_df = pd.read_csv("v.csv")
p_der = pd.read_csv("p_der.csv")
q_der = pd.read_csv("q_der.csv")
batt_charge = pd.read_csv("batt_charge.csv")
batt_discharge = pd.read_csv("batt_discharge.csv")
batt_soc = pd.read_csv("batt_soc.csv")


v_df = v_df.melt(
    id_vars=["id", "name", "t"],
    value_vars=["a", "b", "c"],
    value_name="value",
    var_name="phase",
)
s_df.a = s_df.a.astype(complex)
s_df.b = s_df.b.astype(complex)
s_df.c = s_df.c.astype(complex)
s_df = s_df.melt(
    id_vars=["fb", "tb", "t"],
    value_vars=["a", "b", "c"],
    value_name="complex_value",
    var_name="phase",
)
s_df["p"] = s_df.complex_value.apply(np.real)
s_df["q"] = s_df.complex_value.apply(np.imag)
# px.line_3d(s_df.loc[s_df.phase=="a"], x="t", y="q", z="p", color="tb").write_html("s3d.html")
s_df = s_df.melt(
    id_vars=["fb", "tb", "t", "phase"],
    value_vars=["p", "q"],
    value_name="value",
    var_name="part",
)

p_der = p_der.melt(
    id_vars=["id", "name", "t"],
    value_vars=["a", "b", "c"],
    value_name="value",
    var_name="phase",
)

q_der = q_der.melt(
    id_vars=["id", "name", "t"],
    value_vars=["a", "b", "c"],
    value_name="value",
    var_name="phase",
)
batt_charge = batt_charge.melt(
    id_vars=["id", "name", "t"],
    value_vars=["a", "b", "c"],
    value_name="value",
    var_name="phase",
)
batt_discharge = batt_discharge.melt(
    id_vars=["id", "name", "t"],
    value_vars=["a", "b", "c"],
    value_name="value",
    var_name="phase",
)
batt_soc = batt_soc.melt(
    id_vars=["id", "name", "t"],
    value_vars=["a", "b", "c"],
    value_name="value",
    var_name="phase",
)

px.line(v_df, x="t", y="value", facet_col="phase", color="name").write_html("v.html")
px.line(
    s_df, x="t", y="value", facet_col="phase", facet_row="part", color="tb"
).write_html("s.html")
px.line(p_der, x="t", y="value", facet_col="phase", color="name").write_html(
    "p_der.html"
)
px.line(q_der, x="t", y="value", facet_col="phase", color="name").write_html(
    "q_der.html"
)
px.line(batt_charge, x="t", y="value", facet_col="phase", color="name").write_html(
    "batt_charge.html"
)
px.line(batt_discharge, x="t", y="value", facet_col="phase", color="name").write_html(
    "batt_discharge.html"
)
px.line(batt_soc, x="t", y="value", facet_col="phase", color="name").write_html(
    "batt_soc.html"
)

pass
