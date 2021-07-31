cd gpu-app-collection/src
source setup_environment
cd ~/accel-sim-framework/util/hw_stats
python run_hw.py -B rodinia_2.0-ft -d -N
cd ~
/bin/bash