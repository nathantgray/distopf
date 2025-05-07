from pyomo.environ import (
    Objective,
    minimize,
    value,
    SolverFactory,
    SolverStatus,
    TerminationCondition,
)


def estimate_alpha(model):
    func_obj_est = 0.01 * sum(value(expr) for expr in model.p_L.values())
    fscd_est = estimate_fscd(model)
    alpha = func_obj_est / fscd_est
    return alpha


def estimate_fscd(model):
    Bset = model.Bset
    P_B = model.p_B
    n_C = model.eta_c
    n_D = model.eta_d
    T = model.T

    # (1/η_D[j] - η_C[j]) * P_B_R[j], summed over j in Bset, then multiplied by T
    fscd_terms = [
        (1 / n_D[j, ph] - n_C[j, ph]) * P_B[j, ph] for j in Bset for ph in model.phases
    ]
    fscd_est = sum(fscd_terms) * T
    return fscd_est


def eliminate_scd(model):
    Bset = model.Bset
    Tset = model.Tset
    n_C = model.eta_c
    n_D = model.eta_d
    alpha = estimate_alpha(model)
    print(f"alpha = {alpha}")
    P_c = model.P_c
    P_d = model.P_d

    scd = sum(
        alpha
        * ((1 - n_C[j, ph]) * P_c[t, j, ph] + (1 / n_D[j, ph] - 1) * P_d[t, j, ph])
        for t in Tset
        for j in Bset
        for ph in model.phases
    )
    return scd


def substation_power_minimize(model):
    return sum(model.P_subs[t, ph] for t in model.Tset for ph in model.phases)


def power_flow(model, **kwargs):
    return 0


def loss_minimize(model, **kwargs):
    r = model.r
    # Calculate the total losses
    return sum(
        r[ph + ph][i, j] * (model.P[t, (i, j), ph] ** 2 + model.Q[t, (i, j), ph] ** 2)
        for t in model.Tset
        for (i, j) in model.Lset
        for ph in model.phases
    )


def loss_minimize_with_scd(model, **kwargs):
    r = model.r
    # Calculate the total losses
    total_loss = sum(
        r[ph + ph][i, j] * (model.P[t, (i, j), ph] ** 2 + model.Q[t, (i, j), ph] ** 2)
        for t in model.Tset
        for (i, j) in model.Lset
        for ph in model.phases
    )
    # scd_terms = sum((1 - model.n_c[j,ph]) * model.P_c[t, j,ph] + ((1 / model.n_d[j,ph]) - 1) * model.P_d[t, j,ph] for t in model.Tset for j in model.Bset for ph in model.phases)
    scd_terms = sum(
        (1 - model.eta_c[j, ph]) * model.P_c[t, j, ph]
        + (((1 / model.eta_d[j, ph]) - 1) if model.eta_d[j, ph] != 0 else 1.0)
        * model.P_d[t, j, ph]
        for t in model.Tset
        for j in model.Bset
        for ph in model.phases
    )
    alpha = 1e-3
    return total_loss + alpha * scd_terms


def cost_minimize(model, **kwargs):
    cost = model.cost

    # Compute total substation power for each time period t
    Psubs = {t: sum(model.P_subs[t, ph] for ph in model.phases) for t in model.Tset}

    # Return the total cost across all time periods
    return sum(Psubs[t] * cost[t] for t in model.Tset)


def cost_minimize_with_scd(model, **kwargs):
    cost = model.cost

    # Compute total substation power for each time period t
    Psubs = {t: sum(model.P_subs[t, ph] for ph in model.phases) for t in model.Tset}

    # Return the total cost across all time periods
    total_cost = sum(Psubs[t] * cost[t] for t in model.Tset)
    scd_terms = sum(
        (1 - model.eta_c[j, ph]) * model.P_c[t, j, ph]
        + (((1 / model.eta_d[j, ph]) - 1) if model.eta_d[j, ph] != 0 else 1.0)
        * model.P_d[t, j, ph]
        for t in model.Tset
        for j in model.Bset
        for ph in model.phases
    )
    alpha = 1e-3
    return total_cost + alpha * scd_terms


def pyomo_solve(model, obj_func, **kwargs):
    # Store kwargs as attributes on the model
    for key, value in kwargs.items():
        setattr(model, key, value)
    if hasattr(model, "obj"):
        model.del_component("obj")  # Remove old objective

    # # Get base objective expression
    # base_obj = obj_func(model)  # Call the objective function to get expression
    # # Add SCD terms
    # scd_terms = eliminate_scd(model)
    # full_obj = base_obj + scd_terms
    # # Add the new objective explicitly
    # model.add_component('obj', Objective(expr=full_obj, sense=minimize))
    # model.obj = Objective(rule=full_obj, sense=minimize)
    # Recreate objective with current ADMM terms
    model.obj = Objective(rule=obj_func, sense=minimize)
    opt = SolverFactory("ipopt")
    # opt.options['WarmStart'] = 1
    # opt.options['logfile'] = 'solver_log.txt'
    # opt.options['IISMethod'] = 2
    # opt = SolverFactory('scip', executable=r"C:\Program Files\SCIPOptSuite 9.2.0\bin\scip.exe")
    results = opt.solve(model, tee=False)
    if (
        results.solver.status == "ok"
        and results.solver.termination_condition == "optimal"
    ):
        print("Solver completed successfully.")
    else:
        print(f"Solver failed: {results.solver.termination_condition}")

        # Save IIS to file
        # model.write("model.ilp", format="lp")

    return model
