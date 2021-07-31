import os
import gzip
import copy
import json
import numpy as np
from pandas import DataFrame
import pandas as pd
from scipy import stats
import pickle

# RGS ------------------------------------------------------------
# We open the viz log files, parse the log,
# find the different kernels by finding the points at which
# the cycle count goes to 0, separate them by kernel
# Adapted from Roland Green's code
# Returns a list of logfile paths
def get_paths(directory):
    log_paths = ""
    output_paths = ""
    for root, dirs, files in os.walk(directory):
        for f in files:
            #print(f)
            if f.endswith(".log.gz"):
                log_paths = os.path.join(root, f)
            if ".o" in f:
                output_paths = os.path.join(root, f)
    #print(log_paths + " " + output_paths)
    return log_paths, output_paths

# This is a pre-processed variable containing all the vizlog and output
# files from GPGPUsim AIO.
# The structure of the file is as follows:
# preproc is a dictionary of benchmark applications
# Each application has a kernel_list (associated with the viz log)
# and a kernel_info (associated with the gpgpusim regular output)
# The kernel_list is a list of kernels dictionaries. 
# Each kernel dictionary contains, with a granularity of cycles[0] 
# (usually 500 cycles), cycles, instructions, completed ctas ('ctas')
# and ipcs.
# The kernel_info contains the bigger picture information, like 
# kernel names, total number of ctas spawned per kernel and number of waves
# (i.e. the required number of ctas to fill the GPU resources)
def open_variables(filename):
    proc = {}
    with open('G:/Research/Runs/Run_Aug/'+filename+'.pickle', 'rb') as f:
        proc = pickle.load(f)
    return proc

# Takes in the viz and output path
# Generates a list of kernel dictionaries containing the
# data points' cycle, instructions count, and the number of completed ctas since
# last data point. 
def parse_lines(log_path, output_path):
    # Dump the lines from the file
    with gzip.open(log_path, 'rt') as f:
    # Get the cycles, instructions, and CTAs counts
        cycles = []
        instructions = []
        ctas = []
        for line in f:
            if "globalcyclecount" in line:
                cycles.append(int(line.split(':')[1]))
                continue
            if "globalinsncount" in line:
                instructions.append(int(line.split(':')[1]))
                continue
            if "ctas_completed" in line:
                ctas.append(int(line.split(':')[1]))
                continue
    cycles = np.array(cycles)
    instructions = np.array(instructions)
    ctas = np.array(ctas)
    kernel_indeces = np.where(cycles == cycles[0])
    kernel_indeces = kernel_indeces[0]
    number_of_kernels = len(kernel_indeces)
    kernel_list = []
    for i in range(len(kernel_indeces)-1):
        temp_cycles = cycles[kernel_indeces[i]:kernel_indeces[i+1]]
        #print(temp_cycles)
        temp_ctas = ctas[kernel_indeces[i]:kernel_indeces[i+1]]
        temp_instructions = instructions[kernel_indeces[i]:kernel_indeces[i+1]]
        temp_ipcs = temp_instructions / temp_cycles
        kernel_list.append({'cycles': temp_cycles, 'instructions':temp_instructions, 'ctas':temp_ctas, 'ipcs': temp_ipcs})

    temp_cycles = cycles[kernel_indeces[-1]:]
    temp_ctas = ctas[kernel_indeces[-1]:]
    temp_instructions = instructions[kernel_indeces[-1]:]
    temp_ipcs = temp_instructions / temp_cycles
    kernel_list.append({'cycles': temp_cycles, 'instructions':temp_instructions, 'ctas':temp_ctas, 'ipcs':temp_ipcs})
    
    for kernel in kernel_list:
        if(len(kernel['ctas'])>0): 
            kernel['ctas'][0] = 0
    
    with open(output_path, 'r') as f:
        lines = f.readlines()

    # Get the size of each wave, and total number of CTAs
    wave_sizes = []
    total_ctas = []
    kernel_names = []

    kernel_names, wave_sizes, total_ctas = parse_kernel_info_format_2(output_path)
    
    kernel_info = {
            "kernel_names" : kernel_names,
            "wave_sizes" : wave_sizes,
            "total_ctas" : total_ctas,
            }
    return kernel_list, kernel_info

def parse_kernel_info_format_1(output_path):
    with open(output_path, 'r') as f:
        lines = f.readlines()
    wave_sizes = []
    total_ctas = []
    kernel_names = []
    prev_ctas = 0
    current_kernel = 1
    last_cta_size = 0

    for i in range(len(lines)): 
        if "pushing kernel " in lines[i]:
            i += 2
            if "CTA/core = " in lines[i]:
                wave_temp = int(lines[i].split()[4][:-1])
                last_cta_size = wave_temp
                #wave_sizes.append(int(line.split()[4][:-1]) * 80)
            else:
                wave_temp = last_cta_size
            shaders = 1
            i += 2
            while("bind to kernel" in lines[i]):
                shaders += 1
                i += 1
            wave_sizes.append(wave_temp * shaders)
        if "gpu_tot_issued_cta" in lines[i]:
            issued_ctas = int(lines[i].split()[-1])
            total_ctas.append(issued_ctas - prev_ctas)
            prev_ctas = issued_ctas
            i+=1
        if "kernel_name = " in lines[i]:
            kernel_names.append(lines[i].split()[-1])
            i += 1        
    return kernel_names, wave_sizes, total_ctas

def parse_kernel_info_format_2(output_path):
    with open(output_path, 'r') as f:
        lines = f.readlines()
        
    wave_sizes = []
    total_ctas = []
    kernel_names = []
    prev_ctas = 0
    current_kernel = 1
    last_cta_size = 0
    
    for i in range(len(lines)): 
        if "Reconfigure L1 " in lines[i]:
            #print("Found the line")
            #print(lines[i-1])
            i -= 1
            if "CTA/core = " in lines[i]:
                wave_temp = int(lines[i].split()[4][:-1])
                last_cta_size = wave_temp
                #wave_sizes.append(int(line.split()[4][:-1]) * 80)
            else:
                wave_temp = last_cta_size
            #shaders = 1
            #i += 3
            #while("bind to kernel" in lines[i]):
            #    shaders += 1
            #    i += 2
            wave_sizes.append(wave_temp * 80)
        if "gpu_tot_issued_cta" in lines[i]:
            issued_ctas = int(lines[i].split()[-1])
            total_ctas.append(issued_ctas - prev_ctas)
            prev_ctas = issued_ctas
            i+=1
        if "kernel_name = " in lines[i]:
            kernel_names.append(lines[i].split()[-1])
            i += 1        
    return kernel_names, wave_sizes, total_ctas

# Takes in the kernel_list and kernel_info
# It iterates through each data point per kernel
# Whenever the measured IPC is within 10% of the final IPC,
# and the total number of CTAs completed so far is greater to or equal to than an entire grid (i.e. we want to simulate grids, not partial code)
# we decide to stop sampling there and move on to the next kernel
def RGS_FF(kernel_list, kernel_info, return_completed_ctas = False):
    list_of_kernel_ipcs = [ kernel_list[x]['instructions'][-1] / kernel_list[x]['cycles'][-1] for x in range(len(kernel_list)) ]
    sample_frequency = kernel_list[0]['cycles'][0]
    completed_ctas_returned = 0
    cta_ipcs = []
    for kernel in range(len(kernel_list)):
        #print("Printing kernel: "+str(kernel))
        cycles_to_completion = kernel_list[kernel]['cycles'][-1]
        completed_ctas = 0
        for data_point in range(len(kernel_list[kernel]['ipcs'])):
            # By kernel_ipc we mean the ipc at the actual time of termination
            # By partial_ipc we mean the "instantenous" ipc at time (sample_frequency * n)
            # where n is the sample point in question
            kernel_ipc = list_of_kernel_ipcs[kernel]
            partial_ipc = kernel_list[kernel]['ipcs'][data_point]
            current_cycle = kernel_list[kernel]['cycles'][data_point]
            completed_ctas += kernel_list[kernel]['ctas'][data_point]
            abs_diff = abs(partial_ipc - kernel_ipc) / float(kernel_ipc)
            # In this case we only accept results once the number of CTAs equal at least an entire grid
            if (abs_diff < 0.1) and (completed_ctas > 0):
                cta_ipcs.append({'partial_ipc':partial_ipc, 'cycles':current_cycle, 'cycles_to_completion': cycles_to_completion, 'speedup': cycles_to_completion/current_cycle }) 
                completed_ctas_returned = completed_ctas
                break
            elif (data_point+1 == len(kernel_list[kernel]['ipcs'])):
                completed_ctas_returned = completed_ctas
                cta_ipcs.append({'partial_ipc':partial_ipc, 'cycles':current_cycle, 'cycles_to_completion': cycles_to_completion, 'speedup': cycles_to_completion/current_cycle })
    if(return_completed_ctas):
        return cta_ipcs, completed_ctas_returned
    return cta_ipcs

def Moving_Average(kernel_list, kernel_info, window):
    rolling_means = []
    rolling_stds = []
    for kernel in range(len(kernel_list)):
        df = pd.DataFrame(kernel_list[kernel])
        temp_means = df['ipcs'].rolling(window).mean()
        temp_means.set_axis(df['cycles'],axis=0, inplace=True )
        temp_std = df['ipcs'].rolling(window).std()
        temp_std.set_axis(df['cycles'],axis=0,inplace=True )
        rolling_means.append(temp_means)
        rolling_stds.append(temp_std)
        #print(rolling_std
        #print('['+str(rolling_mean-rolling_std)+','+str(rolling_mean+rolling_std)+']')
    return rolling_means, rolling_stds

def process_all_files(path, return_split = True):
    processed = {}
    for key in path:
        print(key)
        filename = path[key]
        print(filename)
        kl,ki = main(filename)
        ctas_ipcs, completed_ctas = RGS_FF(kl,ki,True)
        overall_ipcs = Overall_IPC(kl,ki)
        processed[key] = {'kernel_list': kl, 'kernel_info':ki, 'RGS_FF_projected_ipcs': ctas_ipcs, 'completed_ctas': completed_ctas, 'overall_ipcs': overall_ipcs}
    return processed

def financial_method(kernel_list, kernel_info, kernel=0):
    series_full = DataFrame(kernel_list[kernel])
    temp_std = series_full['ipcs'].rolling(5).std()
    temp_mean = series_full['ipcs'].rolling(5).mean()
    #temp_std.plot()
    temp_std = temp_std.divide(temp_mean) * 100
    #fig, ax = plt.subplots(figsize=(20, 10))
    #temp_std.plot()
    min_indexes = np.where(temp_std < 0.1)
    try:
        min_index = min_indexes[0][0]
    except:
        min_index = -1
    try:
        model = ARIMA(series_full['ipcs'][:min_index], order=(2,1,0))
        model_fit = model.fit(disp=0)
    #residuals = DataFrame(model_fit.resid)
        output = model_fit.forecast()
        yhat = output[0]
    except:
        output = kernel_list[kernel]['ipcs'][-1]
    return min_index, output
    #temp_mean.plot()
    #line_h = plt.axhline(y=(yhat-output[1]), color="red", linestyle='--',linewidth=5)
    #line_m = plt.axhline(y=yhat, color="purple", linestyle='--',linewidth=3)
    #line_l = plt.axhline(y=(yhat+output[1]),color="yellow", linestyle='--',linewidth=5)

    print(output)
    print("Index: " +str(min_index))

def rolling_mean_method(kernel_list, kernel_info, kernel=0, window=5,threshold=0.1):
    series_full = DataFrame(kernel_list[kernel])
    temp_std = series_full['ipcs'].rolling(window).std()
    temp_mean = series_full['ipcs'].rolling(window).mean()
    temp_std = temp_std.divide(temp_mean) * 100
    min_indexes = np.where(temp_std < threshold)
    try:
        min_index = min_indexes[0][0]
    except:
        min_index = -1
    temp_mean = series_full['ipcs'][:min_index].rolling(window).mean()
    temp_std = series_full['ipcs'][:min_index].rolling(window).std()
    return min_index, [temp_mean.iloc[-1], temp_std, [temp_mean - temp_std, temp_mean + temp_std]]


def rolling_mean_method_with_ctas(kernel_list, kernel_info, kernel=0, window=5,threshold=0.1):
    series_full = DataFrame(kernel_list[kernel])
    temp_std = series_full['ipcs'].rolling(window).std()
    temp_mean = series_full['ipcs'].rolling(window).mean()
    ctas = series_full['ctas'].rolling(window).mean()
    temp_std = temp_std.divide(temp_mean) * 100
    min_indexes_ctas = np.where((temp_std < threshold) & (ctas >= 1.0))
    min_indexes_threshold = np.where((temp_std < threshold * 0.01))
    try:
        min_index_ctas = min_indexes_ctas[0][0]
        min_index_threshold = min_indexes_threshold[0][0]
        min_index = min_index_ctas if min_index_ctas < min_index_threshold else min_index_threshold
    except:
        min_index = -1
    temp_mean = series_full['ipcs'][:min_index].rolling(window).mean()
    temp_std = series_full['ipcs'][:min_index].rolling(window).std()
    return min_index, [temp_mean.iloc[-1], temp_std, [temp_mean - temp_std, temp_mean + temp_std]]

def calculate_remaining_cycles_RGS_version(apps, speedup_info):
    real_cycles_per_app = []
    projected_cycles_per_app = []
    i = 0
    for app in apps:
        real_cycles_per_kernel = []
        projected_cycles_per_kernel = []
        for kernel in range(len(apps[app]['kernel_info']['wave_sizes'])):
            try:
                total_ctas = apps[app]['kernel_info']['total_ctas'][kernel]
                finished_ctas = speedup_info[5][i][kernel]
            except:
                print(speedup_info[4][i])
            cycles_so_far = apps[app]['kernel_list'][kernel]['cycles'][speedup_info[6][i][kernel]]
            projected_cycles_per_kernel.append( (total_ctas-finished_ctas) * cycles_so_far)
        projected_cycles_per_app.append(projected_cycles_per_kernel)
        i+=1
    return projected_cycles_per_app

def calculate_remaining_cycles_SS_cta_version(apps,speedup_info):
    real_cycles_per_app = {}
    projected_cycles_per_app = {}
    i = 0
    for app in apps:
        real_cycles_per_kernel = []
        projected_cycles_per_kernel = []
        for kernel in range(len(apps[app]['kernel_info']['wave_sizes'])):
            try:
                average_instructions_per_CTA = apps[app]['kernel_list'][kernel]['instructions'][-1] / apps[app]['kernel_info']['total_ctas'][kernel]
                total_ctas = apps[app]['kernel_info']['total_ctas'][kernel]
                finished_ctas = speedup_info[5][i][kernel]
                remaining_ctas = total_ctas - finished_ctas
                stable_IPC = speedup_info[3][app][kernel]
                projected_cycles = remaining_ctas * average_instructions_per_CTA / stable_IPC
            except:
                total_ctas = 0
                finished_ctas = 0
                remaining_ctas = 0
                stable_IPC = 0
                projected_cycles = 0
            projected_cycles_per_kernel.append(projected_cycles)
            real_cycles_per_kernel.append(apps[app]['kernel_list'][kernel]['cycles'][-1])
        projected_cycles_per_app[app] = projected_cycles_per_kernel
        real_cycles_per_app[app] = real_cycles_per_kernel
    return projected_cycles_per_app, real_cycles_per_app

def calculate_remaining_cycles_SS_paper_version(apps,speedup_info):
    real_cycles_per_app = {}
    projected_cycles_per_app = {}
    for app in apps:
        real_cycles_per_kernel = []
        projected_cycles_per_kernel = []
        for kernel in range(len(apps[app]['kernel_info']['wave_sizes'])):
            current_instruction_number = speedup_info['cycles_speedup'][app][kernel]
            #if kernel==0:
                #print("Instruction before stopping for kernel "+str(kernel)+" is "+str(current_instruction_number))
            total_instruction_number = speedup_info['cycles_original'][app][kernel]
            remaining_instructions = total_instruction_number - current_instruction_number
            stable_IPC = speedup_info['ipc_projected'][app][kernel][0]
            sample_rate = apps[app]['kernel_list'][kernel]['cycles'][0]
            projected_cycles = (remaining_instructions / stable_IPC) * sample_rate + current_instruction_number
            projected_cycles_per_kernel.append(projected_cycles)
            real_cycles_per_kernel.append(apps[app]['kernel_list'][kernel]['cycles'][-1])
        projected_cycles_per_app[app] = projected_cycles_per_kernel
        real_cycles_per_app[app] = real_cycles_per_kernel
    return projected_cycles_per_app, real_cycles_per_app
#proc = process_all_files(all_paths)

# apps input is the following format  
# 'cycles_original': cycles_original, 'cycles_speedup': cycles_speedup, 
# 'ipc_original': ipc_original, 'ipc_projected': ipc_projected, 
# 'ctas_completed': ctas_completed, 'projected_index': projected_index
def aggregator_plot(apps, method='Financial'):
    speedup = [np.sum(apps['cycles_original'][x]) / np.sum(apps['cycles_speedup'][x]) for x in apps['cycles_original'].keys()]
    geo_mean = scipy.stats.mstats.gmean(speedup)
    fig, ax = plt.subplots(figsize=(140, 20))
    plt.ylim([0,10])
    ax.set_title(method+ ' method - Speedup, Geometric Mean '+str(geo_mean), fontsize=30)
    plt.xticks(fontsize=20,rotation='vertical')
    plt.ylabel("Cycles to Completion / Reduced Cycles ", fontsize=30)
    plt.bar(apps['cycles_original'].keys(), speedup)

    fig, ax = plt.subplots(figsize=(140, 20))
    ax.set_title(method+ ' method - Speedup, Geometric Mean '+str(geo_mean), fontsize=30)
    plt.xticks(fontsize=20,rotation='vertical')
    plt.ylabel("Cycles to Completion / Reduced Cycles ", fontsize=30)
    plt.yscale('log')
    plt.bar(apps['cycles_original'].keys(), speedup)
    
    fig, ax = plt.subplots(figsize=(140, 20))
    plt.xticks(fontsize=20,rotation='vertical')
    ax.set_title(method+ ' method - IPC'+str(geo_mean), fontsize=30)
    average_ipc_real = [np.average(apps['ipc_original'][x]) for x in apps['ipc_original'].keys() ]
    average_ipc_projected = [np.average(apps['ipc_projected'][x][0][0]) for x in apps['ipc_original'].keys() ]
    plt.bar(apps['cycles_original'].keys(),average_ipc_real)
    plt.bar(apps['cycles_original'].keys(),average_ipc_projected, alpha=0.6)
    plt.ylabel("globalInst / globalCycles ",fontsize=30)
    plt.legend(['IPC Running to Completion', 'Projected IPC'],fontsize=30)

        
    fig, ax = plt.subplots(figsize=(140, 20))
    plt.xticks(fontsize=20,rotation='vertical')
    ax.set_title(method+ ' method - IPC'+str(geo_mean), fontsize=30)
    average_ipc_real = [np.average(apps['ipc_original'][x]) for x in apps['ipc_original'].keys() ]
    plt.bar(apps['cycles_original'].keys(),average_ipc_real)
    plt.ylabel("globalInst / globalCycles ",fontsize=30)
    plt.legend(['IPC Running to Completion'],fontsize=30)

    
    fig, ax = plt.subplots(figsize=(140, 20))
    plt.xticks(fontsize=20,rotation='vertical')
    average_error = np.array(average_ipc_real)/np.array(average_ipc_projected)
    ax.set_title(method+ ' method - Error - Average error:'+ str(average_error.mean()), fontsize=30)
    plt.ylabel("Actual IPC / Projected IPC ",fontsize=30)
    #max_ipc_projected = [np.average(apps[3][x][0][2][1]) for x in range(len(apps[3]))]
    #min_ipc_projected = [np.average(apps[3][x][0][2][0]) for x in range(len(apps[3]))]
    plt.bar(apps['cycles_original'].keys(),average_error)
    
    #fig, ax = plt.subplots(figsize=(140, 20))
    #plt.xticks(fontsize=20,rotation='vertical')
    #ipc_projected_per_kernel = {key: [apps['ipc_projected'][key][kernel] for kernel in apps['ipc_projected'][key]] for key in apps['ipc_projected'].keys()}
    #ipc_real_per_kernel = {key: [apps['ipc_original'][key][kernel] for kernel in apps['ipc_original'][key]] for key in apps['ipc_original'].keys()}
    #index_list = []

def projection_speedup_info_dict(apps, method="Financial", window=5, threshold=0.1):
    cycles_original = {}
    cycles_speedup = {}
    ipc_original = {}
    ipc_projected = {}
    remaining_cycles = {}
    ctas_completed = {}
    projected_index = {}
    for app in apps:
        #print(app)
        cycles_per_kernel = []
        ipcs_per_kernel = []
        projected_ipcs_per_kernel = []
        projected_cycles_per_kernel = []
        ctas_completed_at_stop = []
        projected_index_per_kernel = []
        for kernel in range(len(apps[app]['kernel_list'])):
            ctas_per_kernel = DataFrame(apps[app]['kernel_list'][kernel]['ctas'])
            cummulative_ctas_per_kernel = ctas_per_kernel.cumsum()
            cycles_per_kernel.append(apps[app]['kernel_list'][kernel]['cycles'][-1])
            if(method == 'Financial'):
                temp_projected_cycles, temp_projected_ipcs = financial_method(apps[app]['kernel_list'], apps[app]['kernel_info'], kernel)
            elif(method == 'Mean with CTA'):
                temp_projected_cycles, temp_projected_ipcs = rolling_mean_method_with_ctas(apps[app]['kernel_list'], apps[app]['kernel_info'], kernel=kernel, window=window, threshold=threshold)
            else:
                temp_projected_cycles, temp_projected_ipcs = rolling_mean_method(apps[app]['kernel_list'], apps[app]['kernel_info'], kernel=kernel, window=window, threshold=threshold)
            projected_cycles_per_kernel.append(apps[app]['kernel_list'][kernel]['cycles'][temp_projected_cycles])
            projected_index_per_kernel.append(temp_projected_cycles)
            ipcs_per_kernel.append(apps[app]['kernel_list'][kernel]['ipcs'][-1])
            if(temp_projected_cycles == -1):
                projected_ipcs_per_kernel.append([ipcs_per_kernel[-1], 0, [0,0]])
            else:
                projected_ipcs_per_kernel.append(temp_projected_ipcs)
            ctas_completed_at_stop.append(cummulative_ctas_per_kernel.iloc[temp_projected_cycles])
        cycles_original[app] = cycles_per_kernel
        projected_index[app] = projected_index_per_kernel
        cycles_speedup[app] = projected_cycles_per_kernel
        ipc_original[app] = ipcs_per_kernel
        ipc_projected[app] = projected_ipcs_per_kernel
        ctas_completed[app] = ctas_completed_at_stop
    return {'cycles_original': cycles_original, 'cycles_speedup': cycles_speedup, 'ipc_original': ipc_original, 'ipc_projected': ipc_projected, 'ctas_completed': ctas_completed, 'projected_index': projected_index}
