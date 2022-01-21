def kernel_distribution_per_kmean_groups(dataFrame,principalComponents_dataFrame,best_choice=5, kernel_name = 'Kernel Name'):
    #for i in range(starting_from, ending_at):
    histogram_cu = {}
    histogram_fn = {}
    total_runtimes = []
    k_means = KMeans(n_clusters=best_choice, random_state=4).fit(principalComponents_dataFrame)
    dataFrame['Segments'] = k_means.labels_
    print(np.unique(k_means.labels_))
    unique_cufunction_names = dataFrame[kernel_name].unique()
    unique_kernel_names = dataFrame[kernel_name].unique()
    for group in np.unique(k_means.labels_):
        for key in unique_cufunction_names:
            histogram_cu[key] = 0
        for key in unique_kernel_names:
            histogram_fn[key] = 0
        # Extract only the kernels related to the group inside the loop
        fig = plt.figure(figsize = (20,10))
        temp_df = dataFrame.loc[dataFrame['Segments'] == group]
        temp_df.index = range(len(temp_df))
        runtime = temp_df['gpc__cycles_elapsed.avg'].sum()
        total_count = len(temp_df)
        for i in range(len(temp_df)):
            cu_name = temp_df.loc[i, kernel_name]
            func_name = temp_df.loc[i, kernel_name]
            histogram_cu[cu_name] += 1
            histogram_fn[func_name] += 1
        total_runtimes.append(runtime)
        ax = fig.add_subplot(121)
        ax.set_title('K-Means Group '+str(group),fontsize=30)
        chart = plt.bar(histogram_cu.keys(), histogram_cu.values(), 1)
        ax.legend(['Kernel count: '+str(total_count)+'\nTotal Runtime: '+f"{runtime:,}"], fontsize=20)
        ax.set_xticklabels(histogram_cu.keys(),rotation=90)
        ax = fig.add_subplot(122)
        chart = plt.bar(histogram_fn.keys(), histogram_fn.values(), 1)
        ax.set_xticklabels(histogram_fn.keys(),rotation=90)
    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(111)
    ax.set_title("Total Runtimes as a function of Group Id", fontsize=30)
    plt.bar(range(len(total_runtimes)), total_runtimes)
    plt.xlabel('Group ID',fontsize=20)
    plt.ylabel('Run time',fontsize=20)
        #temp_df['CUfunction'].hist()
        
def cycles_rolling_mean_method_with_ctas(kernel_list, kernel_info, kernel='0', window=2,threshold=0.1, cta_wave_threshold = 0.5, debug = False):    
    if (kernel_info[kernel]['wave_sizes'] > kernel_info[kernel]['total_ctas']):
        ctas_threshold = 0.3 * cta_wave_threshold * kernel_info[kernel]['total_ctas']
    else:
        ctas_threshold = cta_wave_threshold * kernel_info[kernel]['wave_sizes']
    if (not (kernel in kernel_list.keys())):
        print("Kernel not found")
        ctas = kernel_info[kernel]['total_ctas']
        cycles_so_far = 0
        total_cycles = 0
        if (kernel not in kernel_list.keys()):
            cycles_so_far = kernel_info[kernel]['total_cycles']
            total_cycles = kernel_info[kernel]['total_cycles']
        else:
            cycles_so_far = kernel_list[kernel]['cycles'][-1]
            total_cycles = kernel_list[kernel]['cycles'][-1]
        if(debug):
            print('Wave much larger than total number of CTAS, skipping')
            print(kernel_info[kernel]['wave_sizes'])
            print(kernel_info[kernel]['total_ctas'])
        error = 0
        speedup = 1
        projection_cycles = cycles_so_far
        cycles_remaining = 0
        total_ctas = ctas
        return {'current_ctas': ctas, 'current_cycles': cycles_so_far, 'total_cycles': total_cycles, 'error': error, 'speedup': speedup,
                      'projection_cycles': total_cycles, 'total_ctas': total_ctas, 'min_index': -1}
    print(kernel_list[kernel].keys())
    try:
        series_full = DataFrame(kernel_list[kernel])
    except:
        series_full = DataFrame({a: kernel_list[kernel][a] for a in ['cycles', 'instructions', 'ctas', 'ipcs']})
    temp_std = series_full['ipcs'].rolling(window).std()
    temp_mean = series_full['ipcs'].rolling(window).mean()
    ctas = np.array([np.sum(kernel_list[kernel]['ctas'][:i]) for i in range(len(kernel_list[kernel]['ctas']))])
    temp_std = temp_std.divide(temp_mean) * 100.0
    
    #ctas_threshold = kernel_info[kernel]  ['wave_sizes'] 
    #plt.figure(figsize=(20,20))
    #plt.plot(range(len(temp_std)), temp_std)
    #plt.hlines(threshold, 0,len(temp_std))
    
    #plt.figure(figsize=(20,20))
    #plt.plot(range(len(ctas)), ctas)
    #plt.hlines(ctas_threshold, 0,len(ctas))
    
    min_indexes_ctas = np.where(ctas >= ctas_threshold)
    min_indexes_threshold = np.where((temp_std < threshold))
    #print("ctas:")
    #print(min_indexes_ctas)
    #print('ipcs:')
    #print(min_indexes_threshold)
    if(debug):
        print("min indexes threshold: " +str(min_indexes_threshold))
    if(len(min_indexes_threshold[0]) == 0 or len(min_indexes_ctas[0]) == 0):
        ctas = kernel_info[kernel]['total_ctas']
        cycles_so_far = kernel_list[kernel]['cycles'][-1]
        total_cycles = kernel_list[kernel]['cycles'][-1]
        error = 0
        speedup = 1
        projection_cycles = kernel_list[kernel]['cycles'][-1]
        cycles_remaining = 0
        total_ctas = ctas
        return {'current_ctas': ctas, 'current_cycles': cycles_so_far, 'total_cycles': total_cycles, 'error': error, 'speedup': speedup,
                      'projection_cycles': total_cycles, 'total_ctas': total_ctas, 'min_index': -1}
    
    min_index_ctas = min_indexes_ctas[0][0]
    min_index_threshold = min_indexes_threshold[0][0]
    min_index = min_index_ctas if min_index_ctas > min_index_threshold else min_index_threshold
    if(debug):
        print("The min index for threshold is " +str(min_indexes_threshold[0][0]))
        print("The chosen min-index for threshold is " +str(min_index))
        print("The number of cycles at minindex is " + str(kernel_list[kernel]['cycles'][min_index]))
    total_ctas = kernel_info[kernel]['total_ctas']
    current_ctas = ctas[min_index]
    cycles_so_far = kernel_list[kernel]['cycles'][min_index]
    total_cycles = kernel_list[kernel]['cycles'][-1]
    ctas_completion_ratio = total_ctas / (current_ctas * 1.0 + 1)
    cycles_remaining = ctas_completion_ratio * cycles_so_far
    error = (cycles_remaining - total_cycles) / (total_cycles * 1.0) * 100.0
    speedup = total_cycles / cycles_so_far 
    if(debug):
        print(cycles_remaining)
        print("CTAs ratio "+ str(ctas_completion_ratio)  )
        print("Total Cycles: " +str(total_cycles))
        print("Current cycles: "+str(cycles_so_far))
        print("Projected cycles remaining: "+str(cycles_remaining))
        print("Actual cycles remaining: "+str(total_cycles - cycles_so_far))
        print("Error: " +str( error ))
        print("Total ctas:" + str(total_ctas))
        print("Current ctas:" + str(current_ctas))
        print("The number of data points at this kernel is " + str(len(kernel_list[kernel]['cycles'])))
    temp_mean = series_full['ipcs'][:min_index].rolling(window).mean()
    temp_std = series_full['ipcs'][:min_index].rolling(window).std()
    return {'current_ctas': ctas[min_index], 'current_cycles': cycles_so_far, 'total_cycles': total_cycles, 'error': error, 'speedup': speedup,
                      'projection_cycles': cycles_remaining, 'total_ctas': total_ctas, 'min_index': min_index}

def projection_cycles_only_selected(kernel_list, kernel_info, PKS_dictionary = None, window=3, threshold=3, cta_wave_threshold=1):
    projection_results = {}
    mean_error = {}
    mean_speedup = {}
    big_total_speedup_numerator = 0.0
    big_total_speedup_denominator = 0.0
    cycles_original_ = []
    cycles_speedup_ = []
    cycles_reduced_ = []
    cycles_projection_ = []
    cycles_error_ = []
    current_ctas_ = []
    total_ctas_ = []
    if(PKS_dictionary):
        kernel_ids = PKS_dictionary[app]['kernel_ids']
    else:
        kernel_ids = kernel_list.keys()
    for kernel in kernel_ids:
            if(PKS_dictionary):
                k = str(int(kernel) + 1)
            else:
                k = kernel
            result_dict = cycles_rolling_mean_method_with_ctas(kernel_list, kernel_info, kernel=k, window=window, threshold=threshold,cta_wave_threshold=cta_wave_threshold,debug=False)
            projection_results[k] = {'cycles_original': result_dict['total_cycles'],
                                          'speedup':result_dict['speedup'], 
                                          'cycles_reduced':result_dict['current_cycles'],
                                          'cycles_projection': result_dict['projection_cycles'],
                                          'error': result_dict['error'], 'current_ctas': result_dict['current_ctas'],
                                          'total_ctas': result_dict['total_ctas'], 'min_index': result_dict['min_index']}
    return projection_results

def projection_cycles(kernel_list, kernel_info, method="Mean with CTA", window=3, threshold=3, cta_wave_threshold = 1):
    projection_results = {}
    mean_error = {}
    mean_speedup = {}
    big_total_speedup_numerator = 0.0
    big_total_speedup_denominator = 0.0
    cycles_original_ = []
    cycles_speedup_ = []
    cycles_reduced_ = []
    cycles_projection_ = []
    cycles_error_ = []
    current_ctas_ = []
    total_ctas_ = []
    #print(sorted(list(kernel_info.keys())))
    for kernel in sorted(list(kernel_info.keys())):
        if(method == 'Financial'):
            pass
        elif(method == 'Mean with CTA'):
            result_dict = cycles_rolling_mean_method_with_ctas(kernel_list, kernel_info, kernel=kernel, window=window, threshold=threshold,cta_wave_threshold=cta_wave_threshold,debug=False)
            projection_results[kernel] = {'cycles_original': result_dict['total_cycles'],
                                          'speedup':result_dict['speedup'], 
                                          'cycles_reduced':result_dict['current_cycles'],
                                          'cycles_projection': result_dict['projection_cycles'],
                                          'error': result_dict['error'], 'current_ctas': result_dict['current_ctas'],
                                          'total_ctas': result_dict['total_ctas']}
        else:
            pass
    return projection_results

import os

def parse_cycles_per_kernel(output_path):
    with open(output_path, 'r') as f:
        cycles_per_kernel = []
        last_kernel_id = []
        result = {}
        for line in f:
            if "kernel id" in line:
                #print(line)
                last_kernel_id.append(line.split()[3])
            if "gpu_sim_cycle" in line:
                #print("found it")
                cycles_per_kernel.append(line.split()[2])
    for i in range(len(cycles_per_kernel)):
        result[last_kernel_id[i]] = cycles_per_kernel[i]
    return result
rawsult = {}
def process_all_files_discrete(RKS_Best_Choices_Dict, paths, single_app = 'BERT'):
    result_dict = {}
    for app in paths:
        output = {}
        output = {}
        output_2 = {}
        cycles_simulator = 0.0
        cycles_simulator_array = []
        best_kernels = RKS_Best_Choices_Dict[app]['kernel_ids']
        group_counts = RKS_Best_Choices_Dict[app]['group_counts']
        print(best_kernels)
        list_of_kernel_cycles = []
        #print(paths[app])
        for file in paths[app]:
            print(file)
            log_path, out_path = RGS.get_paths(file)
            #print(out_path)
            result = parse_cycles_per_kernel(out_path)
            print(log_path)
            print(out_path)
            kl, ki = parse_lines_dictionary(log_path, out_path)
            #print(ki)
            result_2 = projection_cycles(kl, ki,threshold=0.1)
            #print(ki['kernel_ids'])
            output.update(result)
            #print(result_2)
            output_2.update(result_2)
            #print(result)
        aggregate_result = 0.0
        reduced_cycles = 0.0
        projection_pka = 0.0
        projection_pks = 0.0
        for i in range(len(best_kernels)):
            k = str(int(best_kernels[i] + 1))
            #print(output[k])
            #print(group_counts[i])
            try:
                if("resnet_50_64b_4000" in app):
                    print(k)
                    print(output[k])
                cycles_simulator += output_2[k]['cycles_original']
                cycles_simulator_array.append(output_2[k]['cycles_original'])
                reduced_cycles += output_2[k]['cycles_reduced']
                print("For selected kernel " + str(k) + " , the group count is " + str(group_counts[i]))
                print("Simulating to totality takes " + str(output_2[k]['cycles_original']) + 'cycles')
                print("Partially simulating takes " + str(output_2[k]['cycles_reduced']) + 'cycles')
                print("The projected number of cycles is " + str(output_2[k]['cycles_projection']) + 'cycles')
                projection_pka += float(output_2[k]['cycles_projection']) * float(group_counts[i])
                projection_pks += float(output_2[k]['cycles_original']) * float(group_counts[i])
                aggregate_result += float(output_2[k]['cycles_projection']) * float(group_counts[i])
                #cycles_simulator += float(output[k])
            except:
                print(app)
                print(k)
                print('not found')
                pass
        result_dict[app] = {'projected_cycles_pks': projection_pks, 'projected_cycles': aggregate_result, 'reduced_cycles': reduced_cycles, 'simulator_cycles': cycles_simulator, 'simulator_cycles_array': cycles_simulator_array}
        rawsult[app] = output_2
        print(result_dict[app])
    return result_dict

# Tables opened
def process_all_files_discrete_opened_tables(RKS_Best_Choices_Dict, tables, single_app = 'BERT'):
    result_dict = {}
    for app in tables:
        output = {}
        output = {}
        output_2 = {}
        cycles_simulator = 0.0
        cycles_simulator_array = []
        try:
            best_kernels = RKS_Best_Choices_Dict[app]['kernel_ids']
            group_counts = RKS_Best_Choices_Dict[app]['group_counts']
            print(best_kernels)
            list_of_kernel_cycles = []
            kl = tables[app][0]
            ki = tables[app][1]
        except:
            continue
        result_2 = projection_cycles(kl, ki)
            #print(ki['kernel_ids'])
        #output.update(result)
            #print(result_2)
        output_2.update(result_2)
            #print(result)
        aggregate_result = 0.0
        projection_pks = 0.0
        projection_pka = 0.0
        reduced_cycles = 0.0
        total_runtime = 0.0
# (turing_simulations_full_dict['b'][0])['1']['cycles'][-1]
        for kernel in kl:
            total_runtime += kl[kernel]['cycles'][-1]
        for i in range(len(best_kernels)):
            k = str(int(best_kernels[i] + 1))
            #print(output[k])
            #print(group_counts[i])
            try:
                if("resnet_50_64b_4000" in app):
                    print(k)
                    print(output[k])
                cycles_simulator += output_2[k]['cycles_original']
                cycles_simulator_array.append(output_2[k]['cycles_original'])
                reduced_cycles += output_2[k]['cycles_reduced']
                print("For selected kernel " + str(k) + " , the group count is " + str(group_counts[i]))
                print("Simulating to totality takes " + str(output_2[k]['cycles_original']) + ' cycles')
                print("Partially simulating takes " + str(output_2[k]['cycles_reduced']) + ' cycles')
                print("The projected number of cycles is " + str(output_2[k]['cycles_projection']) + ' cycles')
                print("The total runtime is " + str(total_runtime) + " cycles ")
                projection_pka += float(output_2[k]['cycles_projection']) * float(group_counts[i])
                projection_pks += float(output_2[k]['cycles_original']) * float(group_counts[i])
                #cycles_simulator += float(output[k])
            except:
                print(app)
                print(k)
                print('not found')
                pass
        result_dict[app] = {'total_runtime': total_runtime, 'projected_cycles': projection_pka, 'projected_cycles_pka': projection_pka, 'projected_cycles_pks': projection_pks, 'reduced_cycles': reduced_cycles, 'simulator_cycles': cycles_simulator, 'simulator_cycles_array': cycles_simulator_array}
        rawsult[app] = output_2
        print(result_dict[app])
    return result_dict

# Perform PKA on the app
def process_simulation_files(RKS_Best_Choices_Dict, paths, single_app = 'BERT'):
    result_dict = {}
    for app in paths:
        output = {}
        output = {}
        output_2 = {}
        cycles_simulator = 0.0
        cycles_simulator_array = []
        best_kernels = RKS_Best_Choices_Dict[app]['kernel_ids']
        group_counts = RKS_Best_Choices_Dict[app]['group_counts']
#        print(best_kernels)
        list_of_kernel_cycles = []
        #print(paths[app])
#        print(file)
        log_path, out_path = RGS.get_paths(paths[app])
#        print(out_path)
        try:
            result = parse_cycles_per_kernel(out_path)
            kl, ki = parse_lines_dict(log_path, out_path)
            result_2 = projection_cycles(kl, ki)
            output.update(result)
            output_2.update(result_2)
        except Exception as e:
            print(e)
            print(paths[app])
            print(log_path)
            print(out_path)
            continue
        aggregate_result = 0.0
        reduced_cycles = 0.0
        for i in range(len(best_kernels)):
            k = str(int(best_kernels[i] + 1))
            #print(output[k])
            #print(group_counts[i])
            try:
                if("resnet_50_64b_4000" in app):
                    print(k)
                    print(output[k])
                cycles_simulator += output_2[k]['cycles_original']
                cycles_simulator_array.append(output_2[k]['cycles_original'])
                reduced_cycles += output_2[k]['cycles_reduced']
                #print("For selected kernel " + str(k) + " , the group count is " + str(group_counts[i]))
                #print("Simulating to totality takes " + str(output_2[k]['cycles_original']) + 'cycles')
                #print("Partially simulating takes " + str(output_2[k]['cycles_reduced']) + 'cycles')
                #print("The projected number of cycles is " + str(output_2[k]['cycles_projection']) + 'cycles')
                aggregate_result += float(output_2[k]['cycles_projection']) * float(group_counts[i])
                #cycles_simulator += float(output[k])
            except:
                print(app)
                print(k)
                print('not found')
                pass
        result_dict[app] = {'projected_cycles': aggregate_result, 'reduced_cycles': reduced_cycles, 'simulator_cycles': cycles_simulator, 'simulator_cycles_array': cycles_simulator_array}
        rawsult[app] = output_2
        #print(result_dict[app])
    return result_dict