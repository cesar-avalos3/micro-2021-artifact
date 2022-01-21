# This is warp IPC
def hw_ipc_calculate(hw_profile):
    total_cycles = pd.to_numeric(hw_profile['gpc__cycles_elapsed.avg']).sum()
    total_instructions = pd.to_numeric(hw_profile['smsp__inst_executed.sum']).sum()
    return total_instructions / total_cycles

def hw_thread_ipc_calculate(hw_profile):
    cycles = pd.to_numeric(hw_profile['gpc__cycles_elapsed.avg'])
    conversion_factor = pd.to_numeric(hw_profile['smsp__thread_inst_executed_per_inst_executed.ratio'])
    instructions = pd.to_numeric(hw_profile['smsp__inst_executed.sum'])
    total_cycles = 0.0
    total_instructions = 0.0
    for i in range(len(cycles)):
        total_cycles += cycles.loc[i]
        total_instructions += instructions.loc[i] * conversion_factor.loc[i]
    return total_instructions / total_cycles

# This is thread IPC
def sim_ipc_calculate(kernel_list):
    total_cycles = 0.0
    total_instructions = 0.0
    for kernel in kernel_list:
        total_cycles += kernel_list[kernel]['cycles'][-1]
        total_instructions += kernel_list[kernel]['instructions'][-1]
    return total_instructions / total_cycles

# This is thread IPC
def pks_ipc_calculate(kernel_list, sim_table):
    total_instructions = 0.0
    for kernel in kernel_list:
        total_instructions += kernel_list[kernel]['instructions'][-1]
    projected_cycles = sim_table['projected_cycles_pks']
    return total_instructions / projected_cycles

# This is thread IPC
def pka_ipc_calculate(kernel_list, sim_table):
    total_instructions = 0.0
    for kernel in kernel_list:
        total_instructions += kernel_list[kernel]['instructions'][-1]
    projected_cycles = sim_table['projected_cycles_pka']
    return total_instructions / projected_cycles

# This is thread IPC
def pka_ipc_calculate_hw_info(sim_table, hw_profile):
    conversion_factor = pd.to_numeric(hw_profile['smsp__thread_inst_executed_per_inst_executed.ratio'])
    instructions = pd.to_numeric(hw_profile['smsp__inst_executed.sum'])    
    total_instructions = 0.0
    for i in range(len(instructions)):
        total_instructions += instructions.loc[i] * conversion_factor.loc[i]
    projected_cycles = sim_table['projected_cycles_pka']
    return total_instructions / projected_cycles