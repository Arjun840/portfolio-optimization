#!/bin/bash

# Fix Dependency Conflicts on EC2 Instance
# This script resolves the typing-extensions conflict with FastAPI

echo "🔧 Fixing dependency conflicts..."

# Function to fix dependencies
fix_dependencies() {
    echo "📦 Fixing Python package dependencies..."
    
    # Navigate to application directory
    cd /opt/portfolio-backend
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Uninstall conflicting packages
    echo "🗑️  Removing conflicting packages..."
    pip uninstall -y typing-extensions fastapi pydantic starlette
    
    # Install packages in correct order to avoid conflicts
    echo "📥 Installing compatible versions..."
    
    # Install typing-extensions first with correct version
    pip install "typing-extensions>=4.8.0"
    
    # Install core web framework
    pip install "fastapi==0.104.1"
    pip install "uvicorn[standard]==0.23.2"
    pip install "python-multipart==0.0.6"
    
    # Install pydantic and starlette with compatible versions
    pip install "pydantic>=2.4.0"
    pip install "starlette>=0.27.0"
    
    # Install remaining dependencies
    pip install pandas numpy scipy scikit-learn
    pip install yfinance requests
    pip install "python-jose[cryptography]" "passlib[bcrypt]"
    pip install boto3 botocore
    pip install matplotlib plotly
    pip install PyJWT python-dotenv
    
    echo "✅ Dependencies fixed successfully"
}

# Function to restart services
restart_services() {
    echo "🔄 Restarting services..."
    
    # Restart the portfolio backend service
    sudo systemctl restart portfolio-backend
    
    # Wait a moment for startup
    sleep 5
    
    # Check status
    if sudo systemctl is-active --quiet portfolio-backend; then
        echo "✅ Portfolio backend service is running"
        
        # Test if API is responding
        if curl -f -s --connect-timeout 10 "http://localhost:8000/" > /dev/null; then
            echo "✅ API is responding correctly"
        else
            echo "⚠️  Service is running but API not responding yet (normal - may take a minute)"
        fi
    else
        echo "❌ Portfolio backend service failed to start"
        echo "📋 Checking recent logs..."
        sudo journalctl -u portfolio-backend -n 20 --no-pager
        return 1
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "📊 Current Status:"
    echo "   Service: $(sudo systemctl is-active portfolio-backend)"
    echo "   Python: $(cd /opt/portfolio-backend && source venv/bin/activate && python --version)"
    echo "   FastAPI: $(cd /opt/portfolio-backend && source venv/bin/activate && pip show fastapi | grep Version)"
    echo "   typing-extensions: $(cd /opt/portfolio-backend && source venv/bin/activate && pip show typing-extensions | grep Version)"
    echo ""
}

# Main execution
echo "🚀 Starting dependency fix process..."

fix_dependencies
restart_services
show_status

echo "🎉 Dependency fix completed!"
echo ""
echo "🔍 To verify everything is working:"
echo "   sudo systemctl status portfolio-backend"
echo "   curl http://localhost:8000/"
echo ""
echo "📋 To view logs:"
echo "   sudo journalctl -u portfolio-backend -f"
echo "" 