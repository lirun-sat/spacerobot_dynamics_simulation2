import Global_Value
import Set_global_value
import Get_global_value
from rpy2dc import rpy2dc

from Get_global_value import d_time
from Get_global_value import num_q
import numpy as np
import math
from eul2dc import eul2dc
from dc2rpy import dc2rpy
from f_dyn_rk2 import f_dyn_rk2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from calc_aa import calc_aa
from calc_pos import calc_pos
from j_num import j_num
from f_kin_e import f_kin_e


q = np.zeros(num_q)
dq = np.zeros(num_q)
ddq = np.zeros(num_q)

vv = np.zeros((num_q, 3))
ww = np.zeros((num_q, 3))
vd = np.zeros((num_q, 3))
wd = np.zeros((num_q, 3))

v0 = np.array([0, 0, 0])
w0 = np.array([0, 0, 0])
vd0 = np.array([0, 0, 0])
wd0 = np.array([0, 0, 0])

R0 = np.array([0, 0, 0])
Q0 = np.array([0, 0, 0])

A0 = rpy2dc(Q0[0], Q0[1], Q0[2])

print(A0)
# A0 = np.eye(3)

Fe = np.zeros((num_q, 3))
Te = np.zeros((num_q, 3))
F0 = np.array([0, 0, 0])
T0 = np.array([0, 0, 0])

tau = np.zeros(num_q)

Force0 = np.zeros(num_q+6)


'''
qd_tracking = [[], [], [], [], [], []]
d_qd = [[], [], [], [], [], []]
dd_qd = [[], [], [], [], [], []]
'''

qdtempt = []
v00tempt = []
v01tempt = []
v02tempt = []
w0tempt = []

R00_tempt = []
R01_tempt = []
R02_tempt = []

roll_1_tempt = []
pitch1_tempt = []
yaw_1_tempt = []
POS_e1_tempt = []


A0_tempt = []

timetempt = []

# PID parameters set
desired_q = np.array([0.3, 0.2, 0.1, 0.6, 0.5, 0.4])
gain_spring = 10
gain_dumper = 10
# total_steps = 40
total_steps = 12


qd_tracking_0_tempt = []
qd_tracking_1_tempt = []
qd_tracking_2_tempt = []
qd_tracking_3_tempt = []
qd_tracking_4_tempt = []
qd_tracking_5_tempt = []

q0tempt = []
q1tempt = []
q2tempt = []
q3tempt = []
q4tempt = []
q5tempt = []

d_qd_tempt_0 = []
d_qd_tempt_1 = []
d_qd_tempt_2 = []
d_qd_tempt_3 = []
d_qd_tempt_4 = []
d_qd_tempt_5 = []

dq_tempt_0 = []
dq_tempt_1 = []
dq_tempt_2 = []
dq_tempt_3 = []
dq_tempt_4 = []
dq_tempt_5 = []

dd_qd_tempt_0 = []
dd_qd_tempt_1 = []
dd_qd_tempt_2 = []
dd_qd_tempt_3 = []
dd_qd_tempt_4 = []
dd_qd_tempt_5 = []

ddq_tempt_0 = []
ddq_tempt_1 = []
ddq_tempt_2 = []
ddq_tempt_3 = []
ddq_tempt_4 = []
ddq_tempt_5 = []

tau_tempt_0 = []
tau_tempt_1 = []
tau_tempt_2 = []
tau_tempt_3 = []
tau_tempt_4 = []
tau_tempt_5 = []

F0_tempt = []
T0_tempt = []

qd_tracking = np.array(num_q)
d_qd = np.array(num_q)
dd_qd = np.array(num_q)

HH = np.zeros((num_q+6, num_q+6))
dv0 = np.zeros(3)
dw0 = np.zeros(3)

dv0_tempt = []
dw0_tempt = []

K_v = 8
K_p = 8

print('total calculation : %i steps' % (total_steps / d_time))



'''
for t in np.arange(0, total_steps, d_time):
    qd_tracking[0].append(0.6 * math.sin(t))
    qd_tracking[1].append(0.4 * math.sin(t))
    qd_tracking[2].append(0.2 * math.sin(t))
    qd_tracking[3].append(0.1 * math.sin(t))
    qd_tracking[4].append(0.8 * math.cos(t))
    qd_tracking[5].append(0.6 * math.cos(t))


for i in np.arange(1, len(qd_tracking[0]), 1):
    for j in range(0, len(qd_tracking), 1):
        d_qd[j].append((qd_tracking[j][i] - qd_tracking[j][i-1]) / d_time)


for i in np.arange(1, len(d_qd[0]), 1):
    for j in range(0, len(d_qd), 1):
        dd_qd[j].append((d_qd[j][i] - d_qd[j][i-1]) / d_time)
'''


for time in np.arange(0, total_steps, d_time):

    timetempt.append(time)
    if time == 10:
        print('1000 steps of calculation completed! Please wait~')
    elif time == 20:
        print('2000 steps of calculation completed! Please wait~')
    elif time == 30:
        print('3000 steps of calculation completed! Please wait~')
    elif time == 40:
        print('4000 steps of calculation completed! Please wait~')
    elif time == 50:
        print('5000 steps of calculation completed! Please wait~')

    # tau = gain_spring * (desired_q - q) - gain_dumper * dq

    qd_tracking = np.array([0.6 * math.sin(time), 0.4 * math.sin(time), 0.2 * math.sin(time), 0.1 * math.sin(time), 0.8 * math.cos(time), 0.6 * math.cos(time)])

    d_qd = np.array([0.6 * math.cos(time), 0.4 * math.cos(time), 0.2 * math.cos(time), 0.1 * math.cos(time), -0.8 * math.sin(time), -0.6 * math.sin(time)])

    dd_qd = np.array([-0.6 * math.sin(time), -0.4 * math.sin(time), -0.2 * math.sin(time), -0.1 * math.sin(time), -0.8 * math.cos(time), -0.6 * math.cos(time)])

    desire_dv0 = dv0
    desire_dw0 = dw0

    upsilon_phi = dd_qd + K_v * (d_qd - dq) + K_p * (qd_tracking - q)
    # print(upsilon_phi.shape)

    upsilon_base = np.array([desire_dv0, desire_dw0])   # (2,3)
    # print(upsilon_base.flatten().shape)

    upsilon_tempt = np.array([upsilon_base.flatten(), upsilon_phi])
    upsilon = upsilon_tempt.flatten()
    # print(upsilon.shape)

    # print(HH.shape)

    tau = HH @ upsilon + Force0

    R0, A0, v0, w0, q, dq, HH, Force0, dv0, dw0, ddq = f_dyn_rk2(R0, A0, v0, w0, q, dq, tau[0:3], tau[3:6], Fe, Te, tau)

    roll_1, pitch1, yaw_1, roll_2, pitch2, yaw_2 = dc2rpy(A0)

    dv0_tempt.append(dv0)
    dw0_tempt.append(dw0)

    qd_tracking_0_tempt.append(qd_tracking[0])
    qd_tracking_1_tempt.append(qd_tracking[1])
    qd_tracking_2_tempt.append(qd_tracking[2])
    qd_tracking_3_tempt.append(qd_tracking[3])
    qd_tracking_4_tempt.append(qd_tracking[4])
    qd_tracking_5_tempt.append(qd_tracking[5])
    q5tempt.append(q[num_q-1])
    q4tempt.append(q[num_q-2])
    q3tempt.append(q[num_q - 3])
    q2tempt.append(q[num_q - 4])
    q1tempt.append(q[num_q - 5])
    q0tempt.append(q[num_q - 6])

    d_qd_tempt_0.append(d_qd[num_q - 6])
    d_qd_tempt_1.append(d_qd[num_q - 5])
    d_qd_tempt_2.append(d_qd[num_q - 4])
    d_qd_tempt_3.append(d_qd[num_q - 3])
    d_qd_tempt_4.append(d_qd[num_q - 2])
    d_qd_tempt_5.append(d_qd[num_q - 1])
    dq_tempt_0.append(dq[num_q - 6])
    dq_tempt_1.append(dq[num_q - 5])
    dq_tempt_2.append(dq[num_q - 4])
    dq_tempt_3.append(dq[num_q - 3])
    dq_tempt_4.append(dq[num_q - 2])
    dq_tempt_5.append(dq[num_q - 1])

    dd_qd_tempt_0.append(dd_qd[num_q - 6])
    dd_qd_tempt_1.append(dd_qd[num_q - 5])
    dd_qd_tempt_2.append(dd_qd[num_q - 4])
    dd_qd_tempt_3.append(dd_qd[num_q - 3])
    dd_qd_tempt_4.append(dd_qd[num_q - 2])
    dd_qd_tempt_5.append(dd_qd[num_q - 1])
    ddq_tempt_0.append(ddq[num_q - 6])
    ddq_tempt_1.append(ddq[num_q - 5])
    ddq_tempt_2.append(ddq[num_q - 4])
    ddq_tempt_3.append(ddq[num_q - 3])
    ddq_tempt_4.append(ddq[num_q - 2])
    ddq_tempt_5.append(ddq[num_q - 1])

    F0_tempt.append(tau[0:3])
    T0_tempt.append(tau[3:6])

    tau_tempt_0.append(tau[num_q + 0])
    tau_tempt_1.append(tau[num_q + 1])
    tau_tempt_2.append(tau[num_q + 2])
    tau_tempt_3.append(tau[num_q + 3])
    tau_tempt_4.append(tau[num_q + 4])
    tau_tempt_5.append(tau[num_q + 5])

    roll_1_tempt.append(roll_1)
    pitch1_tempt.append(pitch1)
    yaw_1_tempt.append(yaw_1)



# plt.plot(timetempt, POS_e1_tempt, linewidth=1.0, color='red', linestyle='-.', label='POS_e1_tempt')
# plt.plot(timetempt, q1tempt, linewidth=1.0, color='red', linestyle='-.', label='q1tempt')
# plt.plot(timetempt, qdtempt, linewidth=1.0, color='black', linestyle='-.', label='qdtempt')
# plt.plot(timetempt, v00tempt, linewidth=1.0, label='v00tempt')
# plt.plot(timetempt, v01tempt, linewidth=1.0, label='v01tempt')
# plt.plot(timetempt, v02tempt, linewidth=1.0, label='v02tempt')
# plt.plot(timetempt, w0tempt, linewidth=1.0, color='blue', linestyle=':', label='w0tempt')
# plt.plot(timetempt, roll_1_tempt, linewidth=1.0, label='roll_1_tempt')
# plt.plot(timetempt, pitch1_tempt, linewidth=1.0, label='pitch1_tempt')
# plt.plot(timetempt, yaw_1_tempt, linewidth=1.0, label='yaw_1_tempt')
# plt.plot(timetempt, tau_tempt, linewidth=1.0, color='green', linestyle='-', label='tau_tempt')


# plt.plot(timetempt, R00_tempt, linewidth=1.0, label='R00_tempt')
# plt.plot(timetempt, R01_tempt, linewidth=1.0, label='R01_tempt')
# plt.plot(timetempt, R02_tempt, linewidth=1.0, label='R02_tempt')


plt.plot(timetempt, qd_tracking_2_tempt, linewidth=1.0, label='qd_tracking_2_tempt')

plt.plot(timetempt, q2tempt, linewidth=1.0, label='q2tempt')

plt.plot(timetempt, d_qd_tempt_2, linewidth=1.0, label='d_qd_tempt_2')

plt.plot(timetempt, dq_tempt_2, linewidth=1.0, label='dq_tempt_2')

plt.plot(timetempt, dd_qd_tempt_2, linewidth=1.0, label='dd_qd_tempt_2')

plt.plot(timetempt, ddq_tempt_2, linewidth=1.0, label='ddq_tempt_2')

plt.plot(timetempt, tau_tempt_2, linewidth=1.0, label='tau_tempt_2')


# plt.plot(timetempt, tau_tempt_2, linewidth=1.0, label='tau_tempt_2')

# plt.plot(timetempt, q2tempt, linewidth=1.0, label='q2tempt')

# plt.plot(timetempt, q1tempt, linewidth=1.0, label='q1tempt')
# plt.plot(timetempt, q2tempt, linewidth=1.0, label='q2tempt')
# plt.plot(timetempt, q3tempt, linewidth=1.0, label='q3tempt')
# plt.plot(timetempt, q4tempt, linewidth=1.0, label='q4tempt')
# plt.plot(timetempt, q5tempt, linewidth=1.0, label='q5tempt')
# plt.plot(timetempt, q6tempt, linewidth=1.0, label='q6tempt')

plt.legend(loc='upper left')

plt.grid(True)

plt.show()








'''
import test2
import test3
import test4
from test4 import ROOT

print(ROOT)

'''





'''
# use Monte Carlo to generate working space 
from calc_aa import calc_aa
from j_num import j_num
from calc_pos import calc_pos
from Get_global_value import SE
from f_kin_e import f_kin_e


A0 = np.eye(3)
R0 = np.array([0, 0, 0])

# 在[1, 10)之间均匀抽样，数组形状(1,6)
# q = np.random.uniform(-170/180 * math.pi, 170/180 * math.pi, (6,))
# print(q)
i = 0
POS_e1 = np.zeros((2000, 3))
ORI_e1 = np.zeros((2000, 3, 3))
POS_e2 = np.zeros((2000, 3))
ORI_e2 = np.zeros((2000, 3, 3))

while i < 2000:
    # A0 = np.random.rand(3, 3)

###############################################################################################################
    phi = np.random.uniform(-math.pi, math.pi)
    theta = np.random.uniform(-math.pi, math.pi)
    psi = np.random.uniform(-math.pi, math.pi)
    A0 = eul2dc(phi, theta, psi)
###############################################################################################################
    R0 = np.array([0, 0, 0])
    

    q = np.random.uniform(-170 / 180 * math.pi, 170 / 180 * math.pi, (6,))
    AA = calc_aa(A0, q)
    RR = calc_pos(R0, A0, AA, q)

    joints = j_num(0)
    POS_e1[i, :], ORI_e1[i, :, :] = f_kin_e(RR, AA, joints)

    # joints = j_num(1)
    # POS_e2[i, :], ORI_e2[i, :, :] = f_kin_e(RR, AA, joints)

    i += 1


fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(POS_e1[:, 0], POS_e1[:, 1], POS_e1[:, 2])
# ax.scatter(POS_e2[:, 0], POS_e2[:, 1], POS_e2[:, 2])



# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})


##############################################################################################################
X = POS_e1[:, 0]
Y = POS_e1[:, 1]
Z = POS_e1[:, 2]
# ax.scatter(X, Y, Z, 'b-', linewidth=4, label='curve')

null = [6]*len(Z)
ax.scatter(null, Y, Z)
ax.scatter(X, null, Z)
ax.scatter(X, Y, null)
#############################################################################################################



plt.show()
'''






