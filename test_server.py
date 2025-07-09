#!/usr/bin/env python3
"""
Test script to verify the Portfolio Optimization API server can start
"""

import sys
import os
import subprocess
import time
import requests

def test_api_server():
    """Test if the API server can start and respond."""
    print("🔧 Testing Portfolio Optimization API Server...")
    
    # Change to backend directory
    os.chdir('backend')
    
    try:
        # Start the server as a subprocess
        print("📦 Activating virtual environment...")
        process = subprocess.Popen([
            'bash', '-c',
            'source venv/bin/activate && uvicorn api_server:app --host 0.0.0.0 --port 8001'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Test the API
        try:
            response = requests.get('http://localhost:8001/', timeout=5)
            if response.status_code == 200:
                print("✅ API server is working!")
                print("📊 Response:", response.json())
                return True
            else:
                print(f"❌ API returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to API: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    finally:
        # Clean up
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

if __name__ == "__main__":
    success = test_api_server()
    sys.exit(0 if success else 1) 