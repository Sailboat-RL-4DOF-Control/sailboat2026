import numpy as np

current_state = None
current_action = None
state = None
previous_state = None
previous_action = [0, 0]

path = []

# 风场状态
w0 = [5, 0]
var_v = 0.5
var_d = 1/18
w_change_v = 5
w_change_d = 0
w_last_v = w0[0]
w_last_d = w0[1]
w_ini_v = w0[0]
w_ini_d = w0[1]
one_check = 0
two_check = 0
three_check = 0
four_check = 0
total_energy = 0
keep_rudder_clock = 0
keep_sail_clock = 0

obstacle_1 = [0, 0]
obstacle_2 = [0, 0]

obstacle1_set = False
obstacle2_set = False

obstacles = []
observed_obstacles = []

obstacle1_level = 0
has_one_obstacle = False
pos_to_appear = 0


current_time = 0
total_time = 1000


slope = 0
intercept = 0
