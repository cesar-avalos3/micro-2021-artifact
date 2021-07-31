cd gpu-app-collection/src
source setup_environment
cd ~/accel-sim-framework/util/hw_stats
python run_hw.py -B rodinia-3.1 -d -N -C other_stats
cd ~/accel-sim-framework/gpu-simulator/
source setup_environment.sh
cd ../util/job_launching
python run_simulations.py -B rodinia-3.1 -T ../../hw_run/ -C QV100-SASS-VISUAL
cd ~
echo "-----------------------------------------------------------------------"
echo "-      Use Check_Job_Status.sh to check the simulation progress       -"
echo "-      Use Run_PKA.sh to generate Table 2 once the desired number     -"
echo "-      of jobs have finished.                                         -"
echo "-----------------------------------------------------------------------"
#/accel-sim-framework/hw_run/device-0/11.2/
/bin/bash