from Get_global_value import d_time
from Get_global_value import num_q
from calc_forward_dynamics import calc_forward_dynamics
from aw import aw


def calc_dynamics_rk2(A_base, base_position, omega_base_in_body, v_base, q, q_dot, tau):

    q_dot_dot_general = calc_forward_dynamics(A_base, base_position, omega_base_in_body, v_base, q, q_dot, tau)

    omega_base = A_base @ omega_base_in_body
    omega_base_dot_in_inertial_frame = A_base @ q_dot_dot_general[0:3, :]
    v_base_dot = q_dot_dot_general[3:6, :]
    q_dot_dot = q_dot_dot_general[6:6+num_q, :]

    delta1_base_position = d_time * v_base
    delta1_A_base = aw(omega_base) @ A_base - A_base
    delta1_q = d_time * q_dot
    delta1_v_base = d_time * v_base_dot
    delta1_omega_base = d_time * omega_base_dot_in_inertial_frame
    delta1_omega_base_in_body = A_base.T @ delta1_omega_base
    delta1_q_dot = d_time * q_dot_dot

    q_dot_dot_general = calc_forward_dynamics(A_base + delta1_A_base/2, base_position + delta1_base_position/2, omega_base_in_body + delta1_omega_base_in_body/2,
                                              v_base + delta1_v_base/2, q + delta1_q/2, q_dot + delta1_q_dot/2, tau)

    omega_base_dot_in_inertial_frame = A_base @ q_dot_dot_general[0:3, :]
    v_base_dot = q_dot_dot_general[3:6, :]
    q_dot_dot = q_dot_dot_general[6:6 + num_q]

    delta2_base_position = d_time * (v_base + delta1_v_base/2)
    delta2_A_base = aw(omega_base + delta1_omega_base/2) @ A_base - A_base
    delta2_q = d_time * (q_dot + delta1_q_dot/2)
    delta2_v_base = d_time * v_base_dot
    delta2_omega_base = d_time * omega_base_dot_in_inertial_frame
    delta2_omega_base_in_body = A_base.T @ delta2_omega_base
    delta2_q_dot = d_time * q_dot_dot

    q_dot_dot_general = calc_forward_dynamics(A_base + delta2_A_base / 2, base_position + delta2_base_position / 2, omega_base_in_body + delta2_omega_base_in_body / 2,
                                              v_base + delta2_v_base / 2, q + delta2_q / 2, q_dot + delta2_q_dot / 2, tau)

    omega_base_dot_in_inertial_frame = A_base @ q_dot_dot_general[0:3, :]
    v_base_dot = q_dot_dot_general[3:6, :]
    q_dot_dot = q_dot_dot_general[6:6 + num_q, :]

    delta3_base_position = d_time * (v_base + delta2_v_base / 2)
    delta3_A_base = aw(omega_base + delta2_omega_base / 2) @ A_base - A_base
    delta3_q = d_time * (q_dot + delta2_q_dot / 2)
    delta3_v_base = d_time * v_base_dot
    delta3_omega_base = d_time * omega_base_dot_in_inertial_frame
    delta3_omega_base_in_body = A_base.T @ delta3_omega_base
    delta3_q_dot = d_time * q_dot_dot

    q_dot_dot_general = calc_forward_dynamics(A_base + delta3_A_base, base_position + delta3_base_position, omega_base_in_body + delta3_omega_base_in_body,
                                              v_base + delta3_v_base, q + delta3_q, q_dot + delta3_q_dot, tau)

    omega_base_dot_in_inertial_frame = A_base @ q_dot_dot_general[0:3, :]
    v_base_dot = q_dot_dot_general[3:6, :]
    q_dot_dot = q_dot_dot_general[6:6 + num_q]

    delta4_base_position = d_time * (v_base + delta3_v_base)
    delta4_A_base = aw(omega_base + delta3_omega_base) @ A_base - A_base
    delta4_q = d_time * (q_dot + delta3_q_dot)
    delta4_v_base = d_time * v_base_dot
    delta4_omega_base = d_time * omega_base_dot_in_inertial_frame
    delta4_omega_base_in_body = A_base.T @ delta4_omega_base
    delta4_q_dot = d_time * q_dot_dot

    A_base_next = A_base + (delta1_A_base + 2 * delta2_A_base + 2 * delta3_A_base + delta4_A_base) / 6
    base_position_next = base_position + (delta1_base_position + 2 * delta2_base_position + 2 * delta3_base_position + delta4_base_position) / 6
    q_next = q + (delta1_q + 2 * delta2_q + 2 * delta3_q + delta4_q) / 6

    omega_base_in_body_next = omega_base_in_body + (delta1_omega_base_in_body + 2 * delta2_omega_base_in_body + 2 * delta3_omega_base_in_body + delta4_omega_base_in_body) / 6
    v_base_next = v_base + (delta1_v_base + 2 * delta2_v_base + 2 * delta3_v_base + delta4_v_base) / 6
    q_dot_next = q_dot + (delta1_q_dot + 2 * delta2_q_dot + 2 * delta3_q_dot + delta4_q_dot) / 6

    A_base = A_base_next
    base_position = base_position_next
    q = q_next
    omega_base_in_body = omega_base_in_body_next
    v_base = v_base_next
    q_dot = q_dot_next

    return A_base, base_position, omega_base_in_body, v_base, q, q_dot
