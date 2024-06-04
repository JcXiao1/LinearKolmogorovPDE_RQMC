#!/bin/bash
# This script runs 216 tasks in parallel on available GPUs

# List of available GPUs
GPUS=("0" "1" "2" "3" "4")

# Total number of tasks
TOTAL_TASKS=864

# Number of tasks to run concurrently
NUM_CONCURRENT_TASKS=${#GPUS[@]}

# Function to run a task on a specified GPU
run_task() {
    local task_id=$1
    local gpu_id=$2
    echo "Running task $task_id on GPU $gpu_id"
    python hyperparam.py --task_id "$task_id" --gpu "$gpu_id"
}

# Run tasks in parallel on each GPU
for ((i=0; i<TOTAL_TASKS; i+=NUM_CONCURRENT_TASKS)); do
    for ((j=0; j<NUM_CONCURRENT_TASKS; j++)); do
        task_id=$((i + j))
        if ((task_id < TOTAL_TASKS)); then
            gpu_id=${GPUS[j]}
            run_task $task_id $gpu_id &
        fi
    done
    wait  # Wait for the current batch of tasks to complete
done

echo "All tasks completed."