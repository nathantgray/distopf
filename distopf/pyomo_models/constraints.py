from pyomo.environ import ConcreteModel, Constraint, sqrt, inequality  # type: ignore


def active_power_balance_rule(m: ConcreteModel, t, j, ph):
    p_in = sum(m.p[t, (i, j), ph] for (i, jj) in m.link_set if jj == j)
    p_out = sum(m.p[t, (j, k), ph] for (jj, k) in m.link_set if jj == j)
    p_charge_t = m.p_charge[t, j, ph] if j in m.bat_set else 0
    p_discharge_t = m.p_discharge[t, j, ph] if j in m.bat_set else 0
    p_gen_t = m.p_gen[t, j, ph] if j in m.gen_set else 0
    p_load = m.p_load[(t, j, ph)]
    if j in m.swing_bus_set:
        return (
            m.p_subs[t, ph] - p_out - p_load - p_charge_t + p_discharge_t + p_gen_t == 0
        )
    return p_in - p_out - p_load - p_charge_t + p_discharge_t + p_gen_t == 0


def reactive_power_balance_rule(m: ConcreteModel, t, j, ph):
    q_in = sum(m.q[t, (i, j), ph] for (i, jj) in m.link_set if jj == j)
    q_out = sum(m.q[t, (j, k), ph] for (jj, k) in m.link_set if jj == j)
    q_gen_t = m.q_gen[t, j, ph] if j in m.gen_set else 0
    q_load = m.q_load[(t, j, ph)]
    if j in m.swing_bus_set:
        return m.q_subs[t, ph] - q_out - q_load + q_gen_t == 0
    return q_in - q_out - q_load + q_gen_t == 0


def kvl_three_phase_rule(m: ConcreteModel, t, i, j, ph):
    a, b, c = "a", "b", "c"
    aa, ab, ac = "aa", "ab", "ac"
    if ph == "b":
        a, b, c = "b", "c", "a"
        aa, ab, ac = "bb", "bc", "ab"
    if ph == "c":
        a, b, c = "c", "a", "b"
        aa, ab, ac = "cc", "ac", "bc"
    return (
        m.v[t, j, a]
        - m.v[t, i, a]
        + 2 * (m.r[aa][i, j] * m.p[t, i, j, a] + m.x[aa][i, j] * m.q[t, i, j, a])
        + (-m.r[ab][i, j] + sqrt(3) * m.x[ab][i, j]) * m.p[t, i, j, b]
        + (-m.x[ab][i, j] - sqrt(3) * m.r[ab][i, j]) * m.q[t, i, j, b]
        + (-m.r[ac][i, j] - sqrt(3) * m.x[ac][i, j]) * m.p[t, i, j, c]
        + (-m.x[ac][i, j] + sqrt(3) * m.r[ac][i, j]) * m.q[t, i, j, c]
        == 0
    )


def voltage_magnitude_limit_rule(m: ConcreteModel, t, j, ph):
    return inequality(m.v_min[j] ** 2, m.v[t, j, ph], m.v_max[j] ** 2)


def substation_voltage_magnitude_rule(m: ConcreteModel, t, j, ph):
    if j in m.swing_bus_set:
        return m.v[t, j, ph] == m.v_swing[t, j, ph] ** 2
    return Constraint.Skip


def battery_charge_rule(m: ConcreteModel, t, j, ph):
    eta_c = m.charge_efficiency[j, ph]
    eta_d = m.discharge_efficiency[j, ph]
    if eta_d == 0:
        eta_d = 1
    charge_prev = m.charge_state_initial[j, ph]
    if t != m.t_set[0]:
        charge_prev = m.charge_state[t - 1, j, ph]
    charge_next = m.charge_state[t, j, ph]
    delta_charge = m.p_charge[t, j, ph] * eta_c - m.p_discharge[t, j, ph] / eta_d
    return charge_next == charge_prev + delta_charge


def p_gen_rule(m: ConcreteModel, t, j, ph):
    return m.p_gen[t, j, ph] == m.p_gen_setpoint[t, j, ph]


def q_gen_limit_rule(m: ConcreteModel, t, j, ph):
    q_max = sqrt(m.s_gen_rated[j, ph] ** 2 - m.p_gen_setpoint[t, j, ph] ** 2)
    q_min = -q_max
    return inequality(q_min, m.q_gen[t, j, ph], q_max)


def final_soc_rule(m: ConcreteModel, t, j, ph):
    if t == max(m.t_set):
        return m.charge_state[t, j, ph] == m.charge_state_initial[j, ph]
    return Constraint.Skip


def battery_limits_rule(m: ConcreteModel, t, j, ph):
    return inequality(
        m.charge_state_min[j, ph], m.charge_state[t, j, ph], m.charge_state_max[j, ph]
    )


def charging_power_limit_rule(m: ConcreteModel, t, j, ph):
    return m.p_charge[t, j, ph] <= m.p_bat_max[j, ph]


def discharging_power_limit_rule(m: ConcreteModel, t, j, ph):
    return m.p_discharge[t, j, ph] <= m.p_bat_max[j, ph]


def include_lindist_p_flow_constraint(cm: ConcreteModel):
    cm.active_power_balance = Constraint(
        cm.t_set, cm.node_set, cm.phase_set, rule=active_power_balance_rule
    )


def include_lindist_q_flow_constraint(cm: ConcreteModel):
    cm.reactive_power_balance = Constraint(
        cm.t_set, cm.node_set, cm.phase_set, rule=reactive_power_balance_rule
    )


def include_voltage_drop_constraint(cm: ConcreteModel):
    cm.kvl_three_phase = Constraint(
        cm.t_set, cm.link_set, cm.phase_set, rule=kvl_three_phase_rule
    )


def include_voltage_limit_constraint(cm: ConcreteModel):
    cm.voltage_magnitude = Constraint(
        cm.t_set, cm.node_set, cm.phase_set, rule=voltage_magnitude_limit_rule
    )


def include_substation_voltage_constraint(cm: ConcreteModel):
    cm.substation_voltage_magnitude = Constraint(
        cm.t_set, cm.node_set, cm.phase_set, rule=substation_voltage_magnitude_rule
    )


def include_p_gen_constraint(cm: ConcreteModel):
    cm.p_gen_constraint = Constraint(
        cm.t_set, cm.gen_set, cm.phase_set, rule=p_gen_rule
    )


def include_q_gen_limit(cm: ConcreteModel):
    cm.der_reactive_power_limits = Constraint(
        cm.t_set, cm.gen_set, cm.phase_set, rule=q_gen_limit_rule
    )


def include_charge_state_constraint(cm: ConcreteModel):
    cm.battery_dynamics = Constraint(
        cm.t_set, cm.bat_set, cm.phase_set, rule=battery_charge_rule
    )


def include_final_soc_constraint(cm: ConcreteModel):
    cm.final_soc = Constraint(cm.t_set, cm.bat_set, cm.phase_set, rule=final_soc_rule)


def include_soc_limits(cm: ConcreteModel):
    cm.soc_limits = Constraint(
        cm.t_set, cm.bat_set, cm.phase_set, rule=battery_limits_rule
    )


def include_bat_power_limits(cm: ConcreteModel):
    cm.charging_power_limits = Constraint(
        cm.t_set, cm.bat_set, cm.phase_set, rule=charging_power_limit_rule
    )
    cm.discharging_power_limits = Constraint(
        cm.t_set, cm.bat_set, cm.phase_set, rule=discharging_power_limit_rule
    )
