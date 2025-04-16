This is a simple example to check that Orion has been installed correctly and can run.

Please follow the instructions in [INSTALL](INSTALL.md) to start a container with our image.
Then start the Orion process (server and client) by running:
* `cd /root/orion/benchmarking`
* `LD_PRELOAD="/root/orion/src/cuda_capture/libinttemp.so" python launch_jobs.py --algo orion --config_file /root/orion/artifact_evaluation/example/config.json --orion_max_be_duration 1 --orion_start_update 1 --orion_hp_limit 1`
