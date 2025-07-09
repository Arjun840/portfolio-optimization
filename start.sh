#!/bin/bash

echo "============================================================"
echo "🚀 PORTFOLIO OPTIMIZATION SERVER STARTUP"
echo "============================================================"

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the portfolio-optimization directory"
    exit 1
fi

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed. Please install Node.js"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Start backend server
echo "🔧 Starting backend API server..."
cd backend
source venv/bin/activate
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

echo "⏳ Waiting for backend to start..."
sleep 3

# Test if backend is running
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Backend API server started successfully on http://localhost:8000"
else
    echo "❌ Backend server failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend server
echo "🎨 Starting frontend development server..."
cd frontend/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo "============================================================"
echo "✅ ALL SERVERS STARTED SUCCESSFULLY!"
echo ""
echo "📊 Frontend:  http://localhost:5173"
echo "🔧 Backend:   http://localhost:8000"
echo "📚 API Docs:  http://localhost:8000/docs"
echo ""
echo "🛑 Press Ctrl+C to stop all servers"
echo "============================================================"

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up trap to handle Ctrl+C
trap cleanup SIGINT

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID 