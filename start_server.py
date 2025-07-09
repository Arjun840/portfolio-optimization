#!/usr/bin/env python3
"""
Portfolio Optimization Server Startup Script

This script starts both the backend API server and frontend development server.
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

def check_python():
    """Check if Python 3 is available."""
    try:
        result = subprocess.run([sys.executable, '--version'], capture_output=True, text=True)
        print(f"✓ Using Python: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"✗ Python check failed: {e}")
        return False

def install_backend_dependencies():
    """Install backend dependencies."""
    print("📦 Installing backend dependencies...")
    try:
        # Change to backend directory
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        # Activate virtual environment and install dependencies
        activate_script = "venv/bin/activate"
        if not os.path.exists(activate_script):
            print("✗ Virtual environment not found. Please run: python3 -m venv venv")
            return False
            
        # Install API requirements
        subprocess.run([
            "bash", "-c", 
            f"source {activate_script} && pip install -r requirements_api.txt"
        ], check=True)
        
        print("✓ Backend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install backend dependencies: {e}")
        return False

def install_frontend_dependencies():
    """Install frontend dependencies."""
    print("📦 Installing frontend dependencies...")
    try:
        # Change to frontend directory
        frontend_dir = Path(__file__).parent / "frontend" / "frontend"
        os.chdir(frontend_dir)
        
        # Install npm dependencies
        subprocess.run(["npm", "install"], check=True)
        
        print("✓ Frontend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install frontend dependencies: {e}")
        return False
    except FileNotFoundError:
        print("✗ npm not found. Please install Node.js")
        return False

def start_backend():
    """Start the backend API server."""
    print("🚀 Starting backend API server...")
    try:
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
                 # Start uvicorn server
         activate_script = "venv/bin/activate"
         process = subprocess.Popen([
             "bash", "-c",
             f"source {activate_script} && uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload"
         ])
        
        print("✓ Backend API server starting on http://localhost:8000")
        return process
    except Exception as e:
        print(f"✗ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the frontend development server."""
    print("🚀 Starting frontend development server...")
    try:
        frontend_dir = Path(__file__).parent / "frontend" / "frontend"
        os.chdir(frontend_dir)
        
        # Start Vite dev server
        process = subprocess.Popen(["npm", "run", "dev"])
        
        print("✓ Frontend development server starting on http://localhost:5173")
        return process
    except Exception as e:
        print(f"✗ Failed to start frontend: {e}")
        return None

def main():
    """Main function to orchestrate the startup process."""
    print("=" * 60)
    print("🚀 PORTFOLIO OPTIMIZATION SERVER STARTUP")
    print("=" * 60)
    
    # Check Python
    if not check_python():
        sys.exit(1)
    
    # Install dependencies
    if not install_backend_dependencies():
        print("❌ Backend dependency installation failed")
        sys.exit(1)
        
    if not install_frontend_dependencies():
        print("❌ Frontend dependency installation failed")
        sys.exit(1)
    
    # Start servers
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend server")
        sys.exit(1)
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend server")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ ALL SERVERS STARTED SUCCESSFULLY!")
    print("📊 Frontend: http://localhost:5173")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all servers")
    print("=" * 60)
    
    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for processes
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main() 