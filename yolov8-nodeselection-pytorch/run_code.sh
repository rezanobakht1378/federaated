#flwr run . 2>&1 | tee result.txt debug.log
#flwr run . local-simulation-gpu  2>&1 | tee results/run_\$(date +"%Y%m%d_%H%M%S").log

#!/bin/bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
flwr run . local-simulation-gpu  2>&1 | tee "results/run_${TIMESTAMP}.log"
