#!/bin/bash

# Fix Python Installation on EC2 Instance
# This script fixes the Python 3.11 installation issue by using Python 3.10 instead

echo "🔧 Fixing Python installation on EC2 instance..."

# Function to fix Python on existing instance
fix_python_installation() {
    echo "🐍 Installing Python 3.10 (default Ubuntu version)..."
    
    # Update package list
    sudo apt update
    
    # Install Python 3.10 and related packages (default on Ubuntu 22.04)
    sudo apt install -y python3 python3-venv python3-dev python3-pip nginx git curl htop
    
    # Create symbolic link for easier usage
    sudo ln -sf /usr/bin/python3 /usr/bin/python
    
    echo "✅ Python 3.10 installed successfully"
    python3 --version
}

# Function to setup virtual environment and install dependencies
setup_python_environment() {
    echo "🏗️  Setting up Python virtual environment..."
    
    # Navigate to application directory
    cd /opt/portfolio-backend
    
    # Remove old venv if it exists and had issues
    if [ -d "venv" ]; then
        echo "🗑️  Removing old virtual environment..."
        rm -rf venv
    fi
    
    # Create new virtual environment with Python 3.10
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install core dependencies
    echo "📦 Installing dependencies..."
    pip install "fastapi==0.104.1" "uvicorn[standard]==0.23.2"
    pip install pandas numpy scipy scikit-learn
    pip install yfinance requests
    pip install "python-jose[cryptography]" "passlib[bcrypt]"
    
    # Install additional requirements if available
    if [ -f requirements_ec2.txt ]; then
        pip install -r requirements_ec2.txt
    elif [ -f requirements.txt ]; then
        pip install -r requirements.txt
    fi
    
    echo "✅ Python environment setup completed"
}

# Function to restart services
restart_services() {
    echo "🔄 Restarting services..."
    
    # Restart the portfolio backend service
    sudo systemctl restart portfolio-backend
    
    # Check status
    if sudo systemctl is-active --quiet portfolio-backend; then
        echo "✅ Portfolio backend service is running"
    else
        echo "❌ Portfolio backend service failed to start"
        echo "📋 Checking logs..."
        sudo journalctl -u portfolio-backend -n 10
    fi
}

# Main execution
echo "🚀 Starting Python fix process..."

fix_python_installation
setup_python_environment
restart_services

echo ""
echo "🎉 Python fix completed!"
echo ""
echo "🔍 To check if everything is working:"
echo "   sudo systemctl status portfolio-backend"
echo "   sudo journalctl -u portfolio-backend -f"
echo ""
echo "💡 If there are still issues, you may need to reboot:"
echo "   sudo reboot"
echo "" 