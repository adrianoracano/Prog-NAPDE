import math

data_dict = {}
with open('data.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()
step_summation = int(data_dict['step_summation'])
learning_rate = float(data_dict['learning_rate'])
t_max = float(data_dict['t_max'])
beta0 = float(data_dict['beta0'])
K = int(data_dict['K'])
K_val = int(data_dict['K_val'])
N = int(data_dict['N'])
alpha = float(data_dict['alpha'])
training_steps = int(data_dict['training_steps'])
display_step = int(data_dict['display_step'])
temperature_type = data_dict['temperature_type']
TOT = 100
S0 = float(data_dict['S0']) * TOT
I0 = float(data_dict['I0']  ) *  TOT
S_inf = float(data_dict['S_inf']) * TOT
R0 = float(data_dict['R0'])
n_hidden = int(data_dict['n_hidden'])
solver = data_dict['solver']
display_weights = int(data_dict['display_weights'])
mixed = data_dict['mixed']
mixed256 = data_dict['mixed256']
dataset = data_dict['dataset']
tau = float(data_dict['tau'])
n_input = 2
b_ref = alpha * math.log(S0/S_inf) / (TOT - S_inf)
K_test = K_val
dt = 1.0/N


