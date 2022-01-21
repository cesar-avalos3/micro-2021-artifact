import clustering.kmeans
import utils.ipc
import utils.loading
import utils.parsing
import utils.paths
import utils.processing

# List of agnostic features to take into consideration when performing
# the PCA (i.e. only these variables are going to be used)
agnostic_features = ['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
                     'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
                     #'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
                     'smsp__inst_executed.sum','smsp__thread_inst_executed_per_inst_executed.ratio',
                     #'smsp__sass_inst_executed_op_global_atom.sum',
                     'smsp__inst_executed_op_global_ld.sum',
                     'smsp__inst_executed_op_global_st.sum',
                     'smsp__inst_executed_op_shared_ld.sum',
                     'smsp__inst_executed_op_shared_st.sum',
                     #'smsp__inst_executed_op_surface_atom.sum',
                     #'smsp__inst_executed_op_surface_ld.sum',
                     #'smsp__inst_executed_op_surface_red.sum',
                     #'smsp__inst_executed_op_surface_st.sum',
                     'sass__inst_executed_global_loads',
                     'sass__inst_executed_global_stores',
                     'launch__grid_size']

def main():
    errors_ = []
    speedups_ = []

    base_hw_run_directory = '/root/accel-sim-framework/hw_run/device-0/11.2/'
    base_sim_run_directory = '/root/accel-sim-framework/sim_run_11.2/'

    #benchmarks = ['rodinia-3.1', 'deepbench', 'polybench', 'rodinia_2.0-ft', 'parboil', 'cutlass']
    benchmarks = ['rodinia-3.1']
    #benchmarks_volta = ['rodinia-3.1', 'deepbench', 'polybench', 'rodinia_2.0-ft', 'parboil', 'cutlass', 'mlperf']
    benchmarks_volta = ['rodinia-3.1']
    benchmarks_volta_paths = paths.paths(base_hw_run_directory, 'out.csv', benchmarks_volta)
    benchmarks_turing_paths = paths.paths(base_hw_run_directory, 'out.csv', benchmarks)
    benchmarks_ampere_paths = paths.paths(base_hw_run_directory, 'out.csv', benchmarks)

    # ---------- Open HW Profile Data ---------------------------------------------------------------------------------------------------------------- 


    RKS_Table_Volta, List_of_RKS_apps_Volta = open_all(benchmarks_volta_paths)
    RKS_Best_Choices, RKS_List_of_Errors,RKS_List_of_Speedups, RKS_Best_Choices_Dict, HW_Turing = process_all(RKS_Table_Volta, List_of_RKS_apps_Volta, extra_archs=['turing'], all_paths_turing=benchmarks_turing_paths)
    RKS_Best_Choices, RKS_List_of_Errors,RKS_List_of_Speedups, RKS_Best_Choices_Dict, HW_Volta = process_all(RKS_Table_Volta, List_of_RKS_apps_Volta, extra_archs=['turing'], all_paths_turing=benchmarks_volta_paths)

    PKS_list, list_of_errors, list_of_speedups, PKS_dict, extra_results  = process_all(RKS_Table_Volta, List_of_RKS_apps_Volta)

    paths_turing_simulations = paths.paths(base_sim_run_directory, "/RTX2060-SASS-VISUAL/", benchmarks)
    paths_turing_simulations_1B = paths.paths(base_sim_run_directory, "/RTX2060-SASS-VISUAL-1B_INSN/", benchmarks)
    paths_volta_simulations = paths.paths(base_sim_run_directory, "/QV100-SASS-VISUAL/", benchmarks)
    paths_volta_simulations_1B = paths.paths(base_sim_run_directory, "/QV100-SASS-1B_INSN/", benchmarks)

    turing_simulations_full_dict = {}
    turing_simulations_1B_dict = {}
    volta_simulations_full_dict = {}
    volta_simulations_1B_dict = {}

    load_from_scratch = True
    if(load_from_scratch):
        for app in paths_volta_simulations:
            volta_simulations_full_dict[app] = main(paths_volta_simulations[app], output_log_format_type = 2)
            volta_simulations_1B_dict[app] = main(paths_volta_simulations_1B[app], output_log_format_type = 2)

        for app in paths_turing_simulations:
            # main(path, return_format_type = "dict", output_log_format_type = 2):
            turing_simulations_full_dict[app] = main(paths_turing_simulations[app])
            turing_simulations_1B_dict[app] = main(paths_turing_simulations_1B[app])
    else:
        turing_simulations_full_dict = open_variables("Turing-Simulations-Full_dict")
        turing_simulations_1B_dict = open_variables("Turing-Simulations-1B_dict")
        volta_simulations_full_dict = open_variables("Volta-Simulations-Full_dict")
        volta_simulations_1B_dict = open_variables("Volta-Simulations-1B_dict")
        
    turing_SIM = process_all_files_discrete_opened_tables(RKS_Best_Choices_Dict, turing_simulations_full_dict)
    volta_SIM = process_all_files_discrete_opened_tables(RKS_Best_Choices_Dict, volta_simulations_full_dict)

    speedups_geo = []
    TBPoint_InterKernel = {}
    HW_Profile_Tables = {}
    for app in benchmarks_volta_paths:
        print(app)
        try:
            table = open_file(benchmarks_volta_paths[app])
            HW_Profile_Tables[app] = table
        except Exception as e:
            print(e)
            print('error')
            speedups_geo.append(1)
            continue

    # ---------- Open SIM data ------------------------------------------------------------------------------------------

    sim_data = {}
    benchmarks = ['rodinia-3.1', 'deepbench', 'polybench', 'parboil', 'cutlass']
    paths_benchmark = paths.paths(base_sim_run_directory, "/QV100-SASS-VISUAL/", benchmarks)
    for app in paths_benchmark:
        sim_data[app] = main(paths_benchmark[app])

    PKA_dict = {}
    for app in sim_data:
        try:
            print(app)
            PKA_dict[app] = projection_cycles_only_selected(sim_data[app][0], sim_data[app][1], PKS_dict, window=6, threshold=0.25)
        except Exception as e:
            print(e)
            pass


    HW_Turing_Profile_Tables = {}
    HW_Ampere_Profile_Tables = {}

    for app in benchmarks_ampere_paths:
        print(app)
        try:
            table = open_file(benchmarks_ampere_paths[app])
            HW_Ampere_Profile_Tables[app] = table
        except Exception as e:
            #print(e)
            continue

    for app in benchmarks_turing_paths:
        try:
            table = open_file(benchmarks_turing_paths[app])
            HW_Turing_Profile_Tables[app] = table
        except Exception as e:
            #print(e)
            continue

    PKS_Turing_dict = {}

    for app in HW_Turing_Profile_Tables:
        try:
            cycles_projection = np.sum( [ float(HW_Turing_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[int(j)]) * float(PKS_dict[app]['group_counts'][i]) for i,j in enumerate(PKS_dict[app]['kernel_ids']) ] )
            cycles_reduced = np.sum( [ float(HW_Turing_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[int(j)]) for i,j in enumerate(PKS_dict[app]['kernel_ids']) ] )
            cycles_full = np.sum( [float(HW_Turing_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[int(j)]) for j in range(len(HW_Turing_Profile_Tables[app]))] )
            error = np.abs(cycles_full - cycles_projection) / (cycles_projection)
            speedup = cycles_full / cycles_reduced
            PKS_Turing_dict[app] = {'cycles_projection':cycles_projection, 'cycles_full': cycles_full, 'cycles_reduced': cycles_reduced,'speedup': speedup, 'error': error}
        except Exception as e:
            print(e)
            
    PKS_Ampere_dict = {}

    for app in HW_Ampere_Profile_Tables:
        try:
            cycles_projection = np.sum( [ float(HW_Ampere_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[int(j)]) * float(PKS_dict[app]['group_counts'][i]) for i,j in enumerate(PKS_dict[app]['kernel_ids']) ] )
            cycles_reduced = np.sum( [ float(HW_Ampere_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[int(j)]) for i,j in enumerate(PKS_dict[app]['kernel_ids']) ] )
            cycles_full = np.sum( [float(HW_Ampere_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[int(j)]) for j in range(len(HW_Ampere_Profile_Tables[app]))] )
            error = np.abs(cycles_full - cycles_projection) / (cycles_projection)
            speedup = cycles_full / cycles_reduced
            PKS_Ampere_dict[app] = {'cycles_projection':cycles_projection, 'cycles_full': cycles_full, 'cycles_reduced': cycles_reduced,'speedup': speedup, 'error': error}
        except Exception as e:
            print(e)
        
    PKS_Volta_dict = {}

    for app in HW_Profile_Tables:
        try:
            cycles_projection = np.sum( [ float(HW_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[int(j)]) * float(PKS_dict[app]['group_counts'][i]) for i,j in enumerate(PKS_dict[app]['kernel_ids']) ] )
            cycles_reduced = np.sum( [ float(HW_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[int(j)]) for i,j in enumerate(PKS_dict[app]['kernel_ids']) ] )
            cycles_full = np.sum( [float(HW_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[int(j)]) for j in range(len(HW_Profile_Tables[app]))] )
            error = np.abs(cycles_full - cycles_projection) / (cycles_projection)
            speedup = cycles_full / cycles_reduced
            PKS_Volta_dict[app] = {'cycles_projection':cycles_projection, 'cycles_full': cycles_full, 'cycles_reduced': cycles_reduced,'speedup': speedup, 'error': error}
        except Exception as e:
            print(e)

    TBPoint_Complete = {}
    # PKS_dict
    TBPoint_count = 0
    for app in sim_data:
        try:
            reduced_cycles = np.sum([tbpoint[app]['results'][str(int(k+1))]['simulated_cycles'] for k in TBPoint_InterKernel[app]['selected_kernels']])
            full_cycles = np.sum([float(sim_data[app][1][k]['total_cycles']) for k in sim_data[app][1]])
            full_instructions = np.sum([float(sim_data[app][0][k]['instructions'][-1]) for k in sim_data[app][0]])
            ipc_full =  full_instructions / full_cycles
            if('group_instructions' in TBPoint_InterKernel[app]['result']):
                TBPoint_count += 1
                group_instructions = [i / HW_Profile_Tables[app]['smsp__thread_inst_executed_per_inst_executed.ratio'].mean() for i in TBPoint_InterKernel[app]['result']['group_instructions']]
                ipc_projection = np.sum([tbpoint[app]['results'][str(int(k+1))]['projected_ipc'] * (full_instructions / group_instructions[i]) for i,k in enumerate(TBPoint_InterKernel[app]['selected_kernels'])])
            else:
                ipc_projection = np.sum([tbpoint[app]['results'][str(int(k+1))]['projected_ipc'] for i,k in enumerate(TBPoint_InterKernel[app]['selected_kernels'])])
        except Exception as e:
            print(app)
            print(e)
            if(sim_data[app] == []):
                continue
            reduced_cycles = np.sum([float(sim_data[app][1][k]['total_cycles']) for k in sim_data[app][1]])
            full_cycles = np.sum([float(sim_data[app][1][k]['total_cycles']) for k in sim_data[app][1]])
            full_instructions = np.sum([float(sim_data[app][0][k]['instructions'][-1]) for k in sim_data[app][0]])
            ipc_full =  full_instructions / full_cycles
            ipc_projection =  ipc_full
        TBPoint_Complete[app] = {'reduced_cycles': reduced_cycles, 'full_cycles': full_cycles, 'ipc_full': ipc_full, 'ipc_projection': ipc_projection}
        

    PKA_dict_ = PKA_dict
    PKA_Complete = {}
    for app in sim_data:
        try:
            print(PKS_dict[app]['kernel_ids'])
            reduced_cycles = np.sum([float(PKA_dict_[app][k]['cycles_reduced']) for k in PKS_dict[app]['kernel_ids']])
            full_cycles = np.sum([float(sim_data[app][1][k]['total_cycles']) for k in sim_data[app][1]])
            cycles_projection = np.sum([float(PKA_dict_[app][k]['cycles_projection']) * float(PKS_dict[app]['group_counts'][i]) for i,k in enumerate(PKS_dict[app]['kernel_ids'])])
            ipc_full =  np.sum([float(sim_data[app][0][k]['instructions'][-1]) for k in sim_data[app][0]]) / full_cycles
            ipc_projection = np.sum([float(sim_data[app][0][k]['instructions'][-1]) for k in sim_data[app][0]]) / cycles_projection
            if(app == 'polybench-syrk'):
                print('wasap')
        except Exception as e:
            if(sim_data[app] == []):
                continue
            print(e)
            full_cycles = np.sum([float(sim_data[app][1][k]['total_cycles']) for k in sim_data[app][1]])
            reduced_cycles = full_cycles
            cycles_projection = full_cycles
            ipc_full = np.sum([float(sim_data[app][0][k]['instructions'][-1]) for k in sim_data[app][0]]) / full_cycles 
            ipc_projection =  ipc_full
        PKA_Complete[app] =  {'reduced_cycles': reduced_cycles, 'full_cycles': full_cycles, 'projected_cycles': cycles_projection, 'ipc_full': ipc_full, 'ipc_projection': ipc_projection}
        
    Silicon_Volta_Complete = {}
    for app in sim_data:
        try:
            hw_ipc = np.sum([float(HW_Profile_Tables[app]['smsp__thread_inst_executed_per_inst_executed.ratio'].loc[i]) * float(HW_Profile_Tables[app]['smsp__inst_executed.sum'].loc[i]) for i in range(len(HW_Profile_Tables[app]))]) / np.sum([float(HW_Profile_Tables[app]['gpc__cycles_elapsed.avg'].loc[i]) for i in range(len(HW_Profile_Tables[app]))])
        except:
            continue
        Silicon_Volta_Complete[app] = {'ipc': hw_ipc}
        
    Simulations1B_Complete = {}
    for app in sim_data:
        try:
            instructions = np.sum([ Simulations_1B[app][0][k]['instructions'][-1] for k in Simulations_1B[app][0]])
            cycles = np.sum([Simulations_1B[app][0][k]['cycles'][-1] for k in Simulations_1B[app][0]])
            ipc =  instructions / cycles 
        except Exception as e:
            continue
        Simulations1B_Complete[app] = {'ipc': ipc, 'cycles': cycles, 'instructions': instructions }


    save_variables(sim_data, 'simulator_data_10_samples')
    #save_variables(TBPoint_Complete, 'tbpoint_complete')
    save_variables(PKA_Complete, 'pka_complete')
    save_variables(Simulations1B_Complete, '1b_complete')
    #save_variables(TBPoint_InterKernel, 'tbpoint_interkernel')
    #save_variables(tbpoint, 'tbpoint_intrakernel')
    save_variables(PKA_dict, 'pka_intrakernel')
    save_variables(PKS_dict, 'pka_interkernel')
    save_variables(HW_Profile_Tables, 'HW_Profile_Tables')

    save_variables(PKS_Turing_dict, 'PKS_Turing')
    save_variables(PKS_Ampere_dict, 'PKS_Ampere')
    save_variables(PKS_Volta_dict,  'PKS_Volta')