import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pickle
from plotly.validators.scatter.marker import SymbolValidator
from scipy import stats
import paths
import RKS
import RGS

def save_variables(proc, filename):
    with open(filename, 'wb') as f:
        pickle.dump(proc, f)
        
def open_variables(filename):
    with open(filename, 'rb') as f:
        proc = pickle.load(f)
    return proc

PKA_Intra_Kernel = open_variables('pka_intrakernel')
PKA_Inter_Kernel = open_variables('pka_interkernel')
Turing_Simulations_Full = open_variables('Turing-Simulations-Full')
PKS_Turing = open_variables('PKS_Turing')
PKS_Ampere = open_variables('PKS_Ampere')
PKS_Volta = open_variables('PKS_Volta')

HW_Profile_Tables = open_variables("HW_Profile_Tables")

strings = []

strings.append(r'\begin{table*} ')
strings.append('\n')
strings.append(r'\scriptsize')
strings.append('\n')
strings.append(r'\center')
strings.append('\n')
strings.append(r'\caption{Cycle error and Speedup for for \pkstit (PKS) in Silicon and using Accel-Sim. Full Principal Kerenel Analysis (PKA) results also shown for simulation. Entries with "*" do not have data for reasons explained in Section~\ref{sec:eval}. SU=Speedup (in $\times$). Errors are in \%. H=Hours. \label{tab:bigtable}} ')
strings.append('\n')
strings.append(r'\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|}')
strings.append('\n')
strings.append(r'\hline')
strings.append('\n')
strings.append(r'Application & Silicon &  &  &  &  &  & \multicolumn{5}{c|}{Simulation} \\ \hline')
strings.append('\n')
strings.append(r' & \multicolumn{2}{c|}{Volta} & \multicolumn{2}{c|}{Turing} & Ampere &  & \multicolumn{5}{c|}{Volta} \\ \hline')
strings.append('\n')
strings.append(r' & \begin{tabular}[c]{@{}c@{}}Error\\ {[}\%{]}\end{tabular} & SU & \begin{tabular}[c]{@{}c@{}}Error\\ {[}\%{]}\end{tabular} & SU & \begin{tabular}[c]{@{}c@{}}Error\\ {[}\%{]}\end{tabular} & SU & SimError & \begin{tabular}[c]{@{}c@{}}PKS\\ Error\end{tabular} & \begin{tabular}[c]{@{}c@{}}EK\\ SimTime {[}H{]}\\ (SU)\end{tabular} & \begin{tabular}[c]{@{}c@{}}PKA\\ Error\end{tabular} & \begin{tabular}[c]{@{}c@{}}PKA\\ SimTime {[}H{]}\\ (SU)\end{tabular} \\ \hline')
strings.append('\n')
strings.append(r'\textbf{Rodinia Suite} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \hline')
strings.append('\n')

min(len(PKS_Turing),len(PKS_Ampere), len(PKS_Volta))
PKA_Complete = open_variables("PKA_Complete")
PKA_IntraKernel = open_variables('pka_intrakernel')
PKS_InterKernel = open_variables('pka_interkernel')

PKS_Simulation_Cycles = {}
PKS_Simulation_Projected_Cycles = {}
PKA_Simulation_Projected_Cycles = {}
PKA_Simulation_Cycles = {}
for app in PKA_IntraKernel:
    PKS_Simulation_Cycles[app] = np.sum([PKA_IntraKernel[app][k]['cycles_original'] for k in PKA_IntraKernel[app]])
    PKS_Simulation_Projected_Cycles[app] = np.sum([PKA_IntraKernel[app][k]['cycles_original'] * PKS_InterKernel[app]['group_counts'][i] for i,k in enumerate(PKS_InterKernel[app]['kernel_ids'])])
    PKA_Simulation_Projected_Cycles[app] = np.sum([PKA_IntraKernel[app][k]['cycles_projection'] * PKS_InterKernel[app]['group_counts'][i] for i,k in enumerate(PKS_InterKernel[app]['kernel_ids'])])
    PKA_Simulation_Cycles[app] = np.sum([PKA_IntraKernel[app][k]['cycles_reduced'] for k in PKA_IntraKernel[app]])

PKS_Simulation_Error = {}
PKA_Simulation_Error = {}
for app in PKS_Simulation_Cycles:
    cycles_hw_app = pd.to_numeric(HW_Profile_Tables[app]['gpc__cycles_elapsed.avg']).sum()
    PKS_Simulation_Error[app] = round((100 * np.abs(cycles_hw_app - PKS_Simulation_Projected_Cycles[app])/cycles_hw_app),2)
    PKA_Simulation_Error[app] = round((100 * np.abs(cycles_hw_app - PKA_Simulation_Projected_Cycles[app])/cycles_hw_app),2)

PKS_to_PKA_speedup = {}
for app in PKS_Simulation_Cycles:
    PKS_to_PKA_speedup[app] = PKS_Simulation_Cycles / PKA_Simulation_Cycles

rodinia_apps = paths.paths('', '', 'rodinia-3.1')
polybench_apps = paths.paths('', '', 'polybench')
parboil_apps = paths.paths('', '', 'parboil')
deepbench_apps = paths.paths('', '', 'deepbench')
cutlass_apps = paths.paths('', '', 'cutlass')

# Generate rodinia applications

for app in rodinia_apps:
    try:
        app_ = app.replace('_',' ')
        row_string = app_ + ' & '
        row_string += str(round(100 * PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['speedup'],2)) + ' & '
        row_string += str(round(100 * PKS_Turing[app]['error'],2)) + ' & ' + str(round(PKS_Turing[app]['speedup'],2)) + ' & '
        row_string += str(round(100 * PKS_Ampere[app]['error'],2)) + ' & ' + str(round(PKS_Ampere[app]['speedup'],2)) + ' & '
    #Simulation numbers
        row_string += str(round(100 * PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['error'],2)) + ' & '
        row_string += str(round(100 * PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['error'],2)) + ' & '
        row_string += str(round(100 * PKS_Volta[app]['error'],2)) + r" \\ \hline"
        row_string += '\n'
        strings.append(row_string)
    except Exception as e:
        print(e)
        pass

strings.append(r'\textbf{Parboil Suite} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \hline')
strings.append('\n')

for app in parboil_apps:
    try:
        app_ = app.replace('_',' ')
        row_string = app_ + ' & '
        row_string += str(round(100 * PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['speedup'],2)) + ' & '
        row_string += str(round(100 * PKS_Turing[app]['error'],2)) + ' & ' + str(round(PKS_Turing[app]['speedup'],2)) + ' & '
        row_string += str(round(100 * PKS_Ampere[app]['error'],2)) + ' & ' + str(round(PKS_Ampere[app]['speedup'],2)) + ' & '
    #Simulation numbers
        row_string += str(round(PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['error'],2)) + ' & '
        row_string += str(round(PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['error'],2)) + ' & '
        row_string += str(round(PKS_Volta[app]['error'],2)) + r" \\ \hline"
        row_string += '\n'
        strings.append(row_string)
    except:
        pass


strings.append(r'\textbf{Polybench Suite} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \hline')
strings.append('\n')

for app in polybench_apps:
    try:
        app_ = app.replace('_',' ')
        row_string = app_ + ' & '
        row_string += str(round(100 * PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['speedup'],2)) + ' & '
        row_string += str(round(100 * PKS_Turing[app]['error'],2)) + ' & ' + str(round(PKS_Turing[app]['speedup'],2)) + ' & '
        row_string += str(round(100 * PKS_Ampere[app]['error'],2)) + ' & ' + str(round(PKS_Ampere[app]['speedup'],2)) + ' & '
    #Simulation numbers
        row_string += str(round(PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['error'],2)) + ' & '
        row_string += str(round(PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['error'],2)) + ' & '
        row_string += str(round(PKS_Volta[app]['error'],2)) + r" \\ \hline"
        row_string += '\n'
        strings.append(row_string)
    except:
        pass

strings.append(r'\textbf{Cutlass Suite} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \hline')
strings.append('\n')

for app in cutlass_apps:
    try:
        app_ = app.replace('_',' ')
        row_string = app_ + ' & '
        row_string += str(round(100 * PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['speedup'],2)) + ' & '
        row_string += str(round(100 * PKS_Turing[app]['error'],2)) + ' & ' + str(round(PKS_Turing[app]['speedup'],2)) + ' & '
        row_string += str(round(100 * PKS_Ampere[app]['error'],2)) + ' & ' + str(round(PKS_Ampere[app]['speedup'],2)) + ' & '
    #Simulation numbers
        row_string += str(round(PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['error'],2)) + ' & '
        row_string += str(round(PKS_Volta[app]['error'],2)) + ' & ' + str(round(PKS_Volta[app]['error'],2)) + ' & '
        row_string += str(round(PKS_Volta[app]['error'],2)) + r" \\ \hline"
        row_string += '\n'
        strings.append(row_string)
    except:
        pass


strings.append('\n')

strings.append(r'\end{tabular}')
strings.append('\n')
strings.append(r'\end{table*}')
strings.append('\n')

with open("table_2.tex", 'w') as f:
    for line in strings:
        f.write(line)