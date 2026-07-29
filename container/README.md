# Container Build Instructions

## Option 1: Build on Mahti (Recommended)

Since Singularity/Apptainer requires Linux, the easiest way is to build on Mahti:

```bash
# 1. Transfer container files to Mahti
scp -r container/ <user>@mahti.csc.fi:/scratch/<account>/<user>/jwst-mock-lens-simulator/

# 2. SSH to Mahti
ssh <user>@mahti.csc.fi

# 3. Navigate to container directory
cd /scratch/<account>/<user>/jwst-mock-lens-simulator/container

# 4. Load Apptainer module
module load apptainer

# 5. Build container
chmod +x build_on_mahti.sh
./build_on_mahti.sh
```

The container will be built in your scratch directory (e.g., `/scratch/ituomine/gozaliasl/jwst_lens_simulator.sif`).

## Option 2: Build Locally with Docker (Alternative)

If you have Docker and want to create a Docker image instead:

```bash
cd container
docker build -f Dockerfile -t jwst_lens_simulator:latest .
```

Note: This creates a Docker image, not a Singularity container. You'd need to convert it or use Docker on Mahti.

## Option 3: Use Existing Container

If someone has already built the container, you can copy it:

```bash
# From another user
scp <user>@mahti.csc.fi:/scratch/<account>/<user>/jwst_lens_simulator.sif ./
```

## Container Size

The container is approximately 3-5 GB. Make sure you have enough space in your scratch directory.

## Verification

After building, verify the container:

```bash
apptainer exec jwst_lens_simulator.sif python3 -c "import torch; print(torch.__version__)"
apptainer exec jwst_lens_simulator.sif python3 -c "import numpy, pandas, astropy, lenstronomy; print('All packages OK')"
```

