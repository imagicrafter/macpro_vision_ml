#!/bin/bash

# MacBook Pro M4 Vision ML Pipeline - Simple Setup & Startup Script
# Usage: source ./setup.sh (MUST use 'source', not './setup.sh')

set -e  # Exit on any error

echo "🚀 Starting MacBook Pro M4 Vision ML Pipeline"
echo "============================================="

# Check if script is being sourced (required for venv activation to persist)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ ERROR: You must run this script with 'source'"
    echo "   Correct usage: source ./setup.sh"
    echo "   This ensures the virtual environment stays active in your shell"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "requirements.txt" ]] && [[ ! -f "requirements-fixed.txt" ]]; then
    echo "❌ Error: No requirements file found. Run this script from the project root directory."
    return 1
fi

# Setup virtual environment
echo "🔄 Setting up virtual environment..."
if [[ -d ".venv" ]]; then
    echo "✅ Found existing .venv"
    source .venv/bin/activate
else
    echo "📦 Creating new .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
fi

# Verify activation
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✅ Virtual environment active: $(basename $VIRTUAL_ENV)"
else
    echo "❌ Failed to activate virtual environment"
    return 1
fi

# Check if core packages are already installed
echo "🔍 Checking if packages are installed..."
PACKAGES_MISSING=false

python -c "import jupyter" 2>/dev/null || PACKAGES_MISSING=true
python -c "import label_studio" 2>/dev/null || PACKAGES_MISSING=true

if [[ "$PACKAGES_MISSING" == "true" ]]; then
    echo "📦 Installing missing packages..."
    
    # Try requirements-fixed.txt first, fall back to requirements.txt
    if [[ -f "requirements-fixed.txt" ]]; then
        echo "📋 Installing from requirements-fixed.txt..."
        pip install -r requirements-fixed.txt
    elif [[ -f "requirements.txt" ]]; then
        echo "📋 Installing from requirements.txt..."
        echo "⚠️  Note: If you get dependency conflicts, consider using requirements-fixed.txt"
        pip install -r requirements.txt
    fi
else
    echo "✅ Core packages already installed"
fi

# Quick verification of key packages
echo "🔍 Verifying key packages..."
python -c "
import sys
try:
    import jupyter
    print('✅ Jupyter: ready')
except ImportError:
    print('❌ Jupyter: failed')
    sys.exit(1)

try:
    import label_studio
    print('✅ Label Studio: ready')
except ImportError:
    print('❌ Label Studio: failed')
    sys.exit(1)

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError:
    print('❌ PyTorch: failed')

try:
    import ultralytics
    print('✅ YOLO: ready')
except ImportError:
    print('❌ YOLO: failed')
"

# Function to check service status
check_services() {
    echo "🔍 Checking service status..."
    echo ""
    
    # Check Label Studio
    if pgrep -f "label-studio start" >/dev/null 2>&1; then
        LABEL_PIDS=$(pgrep -f "label-studio start")
        echo "🏷️  Label Studio: ✅ RUNNING (PIDs: $LABEL_PIDS)"
        echo "   URL: http://localhost:8080"
    else
        echo "🏷️  Label Studio: ❌ NOT RUNNING"
    fi
    
    # Check Jupyter Lab
    if pgrep -f "jupyter.*lab" >/dev/null 2>&1; then
        JUPYTER_PIDS=$(pgrep -f "jupyter.*lab")
        echo "📓 Jupyter Lab: ✅ RUNNING (PIDs: $JUPYTER_PIDS)"
        echo "   URL: http://localhost:8888"
    else
        echo "📓 Jupyter Lab: ❌ NOT RUNNING"
    fi
    
    # Check virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "🐍 Virtual Env: ✅ ACTIVE ($(basename $VIRTUAL_ENV))"
    else
        echo "🐍 Virtual Env: ❌ NOT ACTIVE"
    fi
    
    echo ""
}

# Function to show help
show_help() {
    echo "🏠 MacBook Pro M4 Vision ML Pipeline - Help"
    echo "=========================================="
    echo ""
    echo "📋 ENVIRONMENT MANAGEMENT:"
    echo "   source ./setup.sh         - Start everything (activate venv + start services)"
    echo "   stop_services             - Stop Label Studio and Jupyter Lab"
    echo "   check_services            - Check what's currently running"
    echo "   show_help                 - Show this help message"
    echo ""
    echo "🔍 STATUS CHECKING:"
    echo "   echo \$VIRTUAL_ENV         - Check if virtual environment is active"
    echo "   pip list                  - See installed packages"
    echo "   ps aux | grep jupyter     - Check if Jupyter is running"
    echo "   ps aux | grep label       - Check if Label Studio is running"
    echo ""
    echo "🌐 ACCESS URLS:"
    echo "   http://localhost:8888     - Jupyter Lab (development)"
    echo "   http://localhost:8080     - Label Studio (annotation)"
    echo ""
    echo "📁 PROJECT STRUCTURE:"
    echo "   .venv/                    - Virtual environment"
    echo "   requirements.txt          - Package dependencies"
    echo "   *.ipynb                   - Jupyter notebooks"
    echo "   label_studio.log          - Label Studio logs"
    echo "   jupyter.log               - Jupyter Lab logs"
    echo ""
    echo "🔧 TROUBLESHOOTING:"
    echo "   tail label_studio.log     - View Label Studio errors"
    echo "   tail jupyter.log          - View Jupyter Lab errors"
    echo "   source .venv/bin/activate - Manually activate environment"
    echo "   deactivate                - Exit virtual environment"
    echo ""
    echo "⚠️  IMPORTANT NOTES:"
    echo "   • Always use 'source ./setup.sh' (not './setup.sh')"
    echo "   • Virtual environment must be active to run services"
    echo "   • Stop services before closing terminal with 'stop_services'"
    echo "   • If ports 8080/8888 are busy, stop other services first"
    echo ""
    echo "🆘 COMMON FIXES:"
    echo "   # If virtual env not active:"
    echo "   source .venv/bin/activate"
    echo ""
    echo "   # If services won't start:"
    echo "   stop_services"
    echo "   pkill -f jupyter"
    echo "   pkill -f label-studio"
    echo "   source ./setup.sh"
    echo ""
    echo "   # If packages missing:"
    echo "   pip install -r requirements.txt"
    echo ""
}

# Function to stop services (define it globally)
stop_services() {
    echo "🛑 Stopping services..."
    
    # More aggressive process killing
    echo "🔍 Finding running processes..."
    
    # Find and kill Label Studio processes
    LABEL_PIDS=$(pgrep -f "label-studio start" 2>/dev/null || true)
    if [[ -n "$LABEL_PIDS" ]]; then
        echo "🏷️  Killing Label Studio processes: $LABEL_PIDS"
        kill -TERM $LABEL_PIDS 2>/dev/null || true
        sleep 2
        # Force kill if still running
        kill -KILL $LABEL_PIDS 2>/dev/null || true
        echo "✅ Label Studio stopped"
    else
        echo "ℹ️  Label Studio not running"
    fi
    
    # Find and kill Jupyter processes
    JUPYTER_PIDS=$(pgrep -f "jupyter.*lab" 2>/dev/null || true)
    if [[ -n "$JUPYTER_PIDS" ]]; then
        echo "📓 Killing Jupyter Lab processes: $JUPYTER_PIDS"
        kill -TERM $JUPYTER_PIDS 2>/dev/null || true
        sleep 2
        # Force kill if still running
        kill -KILL $JUPYTER_PIDS 2>/dev/null || true
        echo "✅ Jupyter Lab stopped"
    else
        echo "ℹ️  Jupyter Lab not running"
    fi
    
    # Clean up PID files
    rm -f .label_studio_pid .jupyter_pid
    
    # Verify they're actually stopped
    sleep 1
    if pgrep -f "label-studio start" >/dev/null 2>&1; then
        echo "⚠️  Warning: Label Studio may still be running"
        echo "   Try: sudo pkill -f label-studio"
    fi
    
    if pgrep -f "jupyter.*lab" >/dev/null 2>&1; then
        echo "⚠️  Warning: Jupyter Lab may still be running"
        echo "   Try: sudo pkill -f jupyter"
    fi
    
    echo "🛑 Stop services command completed"
}

# Stop any existing services first
echo "🧹 Stopping any existing services..."
stop_services
sleep 2

# Start Label Studio
echo "🏷️  Starting Label Studio on port 8080..."
nohup label-studio start --port 8080 > label_studio.log 2>&1 &
LABEL_STUDIO_PID=$!
echo "$LABEL_STUDIO_PID" > .label_studio_pid

# Start Jupyter Lab  
echo "📓 Starting Jupyter Lab on port 8888..."
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser > jupyter.log 2>&1 &
JUPYTER_PID=$!
echo "$JUPYTER_PID" > .jupyter_pid

# Give services time to start
echo "⏳ Waiting for services to start..."
sleep 5

# Check if services started and show errors immediately
echo "🔍 Checking service status..."

if kill -0 $LABEL_STUDIO_PID 2>/dev/null; then
    echo "✅ Label Studio started (PID: $LABEL_STUDIO_PID)"
else
    echo "❌ Label Studio failed to start"
    echo "📄 Label Studio error log:"
    echo "=========================="
    tail -10 label_studio.log 2>/dev/null || echo "No log file found"
    echo "=========================="
fi

if kill -0 $JUPYTER_PID 2>/dev/null; then
    echo "✅ Jupyter Lab started (PID: $JUPYTER_PID)"
else
    echo "❌ Jupyter Lab failed to start"
    echo "📄 Jupyter error log:"
    echo "===================="
    tail -10 jupyter.log 2>/dev/null || echo "No log file found"
    echo "===================="
fi

echo ""
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo "🌐 Label Studio:  http://localhost:8080"
echo "🌐 Jupyter Lab:   http://localhost:8888"
echo ""
echo "📋 AVAILABLE COMMANDS:"
echo "   stop_services   - Stop both services"
echo "   check_services  - Check what's running"
echo "   show_help       - Show detailed help"
echo ""
echo "✨ Virtual environment is active in this shell"
echo ""
echo "💡 If services failed to start, check the logs above or run:"
echo "   tail label_studio.log"
echo "   tail jupyter.log"