#!/usr/bin/env python3
"""
Simple Backend Server Startup Script for Portfolio Optimization API
"""

import subprocess
import sys
import os
from pathlib import Path

def start_backend():
    """Start only the backend API server."""
    print("=" * 60)
    print("🚀 STARTING PORTFOLIO OPTIMIZATION API SERVER")
    print("=" * 60)
    
    try:
        # Change to backend directory
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        # Check if virtual environment exists
        if not os.path.exists("venv/bin/activate"):
            print("❌ Virtual environment not found in backend/venv/")
            print("Please run: cd backend && python3 -m venv venv")
            return False
        
        print("📦 Activating virtual environment...")
        print("🚀 Starting API server on http://localhost:8000")
        print("📚 API documentation will be available at http://localhost:8000/docs")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the server using uvicorn directly
        subprocess.run([
            "bash", "-c",
            "source venv/bin/activate && uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

if __name__ == "__main__":
    start_backend() 