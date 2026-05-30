#!/bin/bash
# Run the notebook non-interactively (restarts kernel automatically)
# Usage: bash run_notebook.sh

export PATH="$PATH:$HOME/Library/Python/3.13/bin"

cd "$(dirname "$0")"

echo "Running ieeecis_exploration.ipynb..."
echo "This will take 15-30 minutes depending on your machine."
echo ""

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    ieeecis_exploration.ipynb \
    --output ieeecis_exploration_executed.ipynb

echo ""
echo "Done! Output saved to ieeecis_exploration_executed.ipynb"
