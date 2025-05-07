from pyomo.environ import *
import numpy as np


def store_results(model):
    modelVals = {}

    # Initialize containers for each variable
    modelVals["P_subs"] = {}
    modelVals["Q_subs"] = {}
    modelVals["P"] = {}
    modelVals["Q"] = {}
    modelVals["v"] = {}
    modelVals["p_D"] = {}
    modelVals["q_D"] = {}
    modelVals["P_c"] = {}
    modelVals["P_d"] = {}
    modelVals["B"] = {}

    # Store the optimization results for each variable
    for t in model.Tset:
        for ph in model.phases:
            modelVals["P_subs"][(t, ph)] = value(model.P_subs[t, ph])

    for t in model.Tset:
        for ph in model.phases:
            modelVals["Q_subs"][(t, ph)] = value(model.Q_subs[t, ph])

    for t in model.Tset:
        for i, j in model.Lset:
            for ph in model.phases:
                modelVals["P"][(t, (i, j), ph)] = value(model.P[t, (i, j), ph])
                modelVals["Q"][(t, (i, j), ph)] = value(model.Q[t, (i, j), ph])

    for t in model.Tset:
        for j in model.Nset:
            for ph in model.phases:
                modelVals["v"][(t, j, ph)] = value(model.v[t, j, ph])

    for t in model.Tset:
        for j in model.Dset:
            for ph in model.phases:
                modelVals["p_D"][(t, j, ph)] = value(model.p_D[t, j, ph])
                modelVals["q_D"][(t, j, ph)] = value(model.q_D[t, j, ph])

    for t in model.Tset:
        for j in model.Bset:
            for ph in model.phases:
                modelVals["P_c"][(t, j, ph)] = value(model.P_c[t, j, ph])
                modelVals["P_d"][(t, j, ph)] = value(model.P_d[t, j, ph])
                modelVals["B"][(t, j, ph)] = value(model.B[t, j, ph])

    modelVals["objective_value"] = value(model.obj)

    return modelVals
