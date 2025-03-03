#!/bin/bash
# Setup script for installing dependencies using UV package manager

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV package manager not found. Installing UV..."
    curl -fsSL https://astral.sh/uv/install.sh | bash
    # Add UV to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Setup complete! You can now run the script with:"
echo "python generate_comprehensive_report.py"
echo ""
echo "To generate just the summary reports:"
echo "python generate_comprehensive_report.py --summary-only" 