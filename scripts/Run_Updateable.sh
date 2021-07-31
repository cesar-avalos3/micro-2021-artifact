cd gpu-app-collection/src
source setup_environment
cd ~/accel-sim-framework/util/hw_stats
python run_hw.py -B rodinia-3.1 -d -N -C other_stats
# To run a more complete suite, comment the line above and uncomment the line below 
# python run_hw.py -B rodinia-3.1,polybench,parboil -d -N -C other_stats
# Refer to accel-sim-framework/util/job_launching/apps/define-all-apps.yaml for the list of available benchmarks
cd ~/accel-sim-framework/gpu-simulator/
source setup_environment.sh
cd ../util/job_launching
python run_simulations.py -B rodinia-3.1 -T ../../hw_run/ -C QV100-SASS-VISUAL
# python run_simulations.py -B rodinia-3.1,polybench,parboil -T ../../hw_run/ -C QV100-SASS-VISUAL
cd ~
echo "-----------------------------------------------------------------------"
echo "-      Use Check_Job_Status.sh to check the simulation progress       -"
echo "-      Use Run_PKA.sh to generate Table 2 once the desired number     -"
echo "-      of jobs have finished.                                         -"
echo "-----------------------------------------------------------------------"
#/accel-sim-framework/hw_run/device-0/11.2/
/bin/bash