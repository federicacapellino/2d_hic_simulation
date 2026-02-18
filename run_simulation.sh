#!/bin/bash

# Script to run Fluidum simulation with SLURM parameters from YAML config

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check arguments.
# Supported syntaxes:
#   ./run_simulation.sh <config.yaml>
#   ./run_simulation.sh <julia_script.jl> <config.yaml> [extra_julia_args...]
# If a Julia script (ends with .jl) is provided as first arg, treat second arg as the YAML.
if [ $# -lt 1 ]; then
    echo "❌ Error: No config YAML file provided"
    echo ""
    echo "Usage: $0 [julia_script.jl] <config.yaml> [extra_julia_args...]"
    echo ""
    echo "Example:"
    echo "  sbatch $0 config.yaml"
    echo "  sbatch $0 collision_event.jl config.yaml"
    exit 1
fi

# Default Julia script to run (keeps backward compatibility)
JULIA_SCRIPT="collision_event_yaml.jl"

if [[ "$1" == *.jl ]] && [ $# -ge 2 ]; then
    JULIA_SCRIPT="$1"
    CONFIG_YAML="$2"
    shift 2
else
    CONFIG_YAML="$1"
    shift 1
fi

# Remaining arguments (if any) are forwarded to Julia
JULIA_EXTRA_ARGS=("$@")

# Check if the YAML file exists
if [ ! -f "$CONFIG_YAML" ]; then
    echo "❌ Error: Config file not found: $CONFIG_YAML"
    exit 1
fi

# Convert to absolute path if needed
if [[ "$CONFIG_YAML" != /* ]]; then
    CONFIG_YAML="$(cd "$(dirname "$CONFIG_YAML")" && pwd)/$(basename "$CONFIG_YAML")"
fi

# Load YAML parsing utility
# Simple YAML parser function
parse_yaml() {
    local yaml_file=$1
    
    # Use python to parse YAML
    python3 << 'EOF'
import yaml, shlex

with open('$yaml_file', 'r') as f:
    config = yaml.safe_load(f)

slurm = config.get('slurm', {})

def q(x):
    return shlex.quote(str(x)) if x is not None else "''"

print(f"MAX_TIME={q(slurm.get('max_time'))}")
print(f"PARTITION={q(slurm.get('partition'))}")
print(f"WORK_DIR={q(slurm.get('work_directory'))}")
print(f"CPUS_PER_JOB={q(slurm.get('cpus_per_job'))}")
print(f"MEMORY={q(slurm.get('memory'))}")
print(f"NUM_JOBS={q(slurm.get('number_of_jobs'))}")
print(f"MAX_CONCURRENT={q(slurm.get('max_concurrent_jobs'))}")
EOF
}

# Parse the provided YAML file
eval "$(parse_yaml "$CONFIG_YAML")"

# Validate that required SLURM parameters were provided in the YAML
required_vars=(MAX_TIME PARTITION WORK_DIR CPUS_PER_JOB MEMORY NUM_JOBS MAX_CONCURRENT)
for v in "${required_vars[@]}"; do
    if [ -z "${!v}" ] || [ "${!v}" = "''" ]; then
        echo "❌ Error: SLURM parameter $v is missing in $CONFIG_YAML"
        exit 1
    fi
done

echo ""

# Check if SLURM is available
if ! command -v sbatch &> /dev/null; then
    echo "⚠️  SLURM not found - running locally without SLURM"
    echo ""
    echo "Starting Julia simulation with config: $CONFIG_YAML"
    julia "$SCRIPT_DIR/collision_event_yaml.jl" "$CONFIG_YAML"
else
    echo "========================================"
    echo "  🌊 FLUIDUM Simulation Launcher 🌊"
    echo "========================================"
    echo "📋 Configuration file: $CONFIG_YAML"
    echo "📋 SLURM Parameters extracted:"
    echo "  ├─ Max time:          $MAX_TIME minutes"
    echo "  ├─ Partition:         $PARTITION"
    echo "  ├─ CPUs per job:      $CPUS_PER_JOB"
    echo "  ├─ Memory:            $MEMORY"
    echo "  ├─ Number of jobs:    $NUM_JOBS"
    echo "  ├─ Max concurrent:    $MAX_CONCURRENT"
    echo "  └─ Work directory:    $WORK_DIR"
    echo ""
    
    # Create temporary SLURM job script
    JOB_SCRIPT="$SCRIPT_DIR/fluidum_job.slurm"
    
    # Prepare quoted variables for safe embedding into the job script
    if [[ "$JULIA_SCRIPT" == /* ]] || [[ "$JULIA_SCRIPT" == */* ]]; then
        JULIA_CMD="$JULIA_SCRIPT"
    else
        JULIA_CMD="$SCRIPT_DIR/$JULIA_SCRIPT"
    fi
    JULIA_CMD_Q=$(printf '%q' "$JULIA_CMD")
    CONFIG_YAML_Q=$(printf '%q' "$CONFIG_YAML")
    JULIA_EXTRA_Q=""
    for a in "${JULIA_EXTRA_ARGS[@]}"; do
        JULIA_EXTRA_Q+=" $(printf '%q' "$a")"
    done

    cat > "$JOB_SCRIPT" << SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=fluidum-sim
#SBATCH --time=$MAX_TIME
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS_PER_JOB
#SBATCH --mem=$MEMORY
#SBATCH --array=1-$NUM_JOBS%$MAX_CONCURRENT
#SBATCH --chdir=$WORK_DIR

echo "Running FLUIDUM simulation job \$SLURM_ARRAY_TASK_ID of $NUM_JOBS"
echo "Job started at \$(date)"
echo "Running on host \$(hostname)"
echo "CPUs allocated: \$SLURM_CPUS_PER_TASK"
echo ""

cd "$SCRIPT_DIR"
echo "Running: julia $JULIA_CMD_Q $CONFIG_YAML_Q$JULIA_EXTRA_Q"
julia $JULIA_CMD_Q $CONFIG_YAML_Q$JULIA_EXTRA_Q

echo ""
echo "Job completed at \$(date)"
SLURM_SCRIPT

    chmod +x "$JOB_SCRIPT"
    
    echo "📝 SLURM script created: $JOB_SCRIPT"
    echo ""
    echo "Submitting job to SLURM..."
    echo ""
    
    # Submit the job
    sbatch "$JOB_SCRIPT"
    
    echo ""
    echo "✅ Job submitted! Check job status with: squeue -u $USER"
fi

echo ""
echo "========================================"
