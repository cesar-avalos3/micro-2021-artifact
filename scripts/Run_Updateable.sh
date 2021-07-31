cd gpu-app-collection/src
source setup_environment
cd ~/accel-sim-framework/util/hw_stats
python run_hw.py -B rodinia_2.0-ft -d -N -C other_stats
cd ~/accel-sim-framework/gpu-simulator/
source setup_environment.sh
cd ../util/job_launching
python run_simulations.py -B rodinia_2.0-ft -T ../../hw_run/ -C QV100-SASS-VISUAL

#/accel-sim-framework/hw_run/device-0/11.2/
/bin/bash