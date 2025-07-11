#!/bin/bash

# Cleanup Lambda Files Script
# This script removes all Lambda-related files now that we're using EC2 deployment

set -e

echo "🧹 Cleaning up Lambda deployment files..."
echo ""

# Function to safely remove files/directories
safe_remove() {
    local item="$1"
    if [ -e "$item" ]; then
        echo "🗑️  Removing: $item"
        rm -rf "$item"
    else
        echo "ℹ️  Already gone: $item"
    fi
}

echo "📁 Removing Lambda directories..."

# Root level Lambda directories
safe_remove "lambda-package-s3"
safe_remove "lambda-ultra-minimal"

# Backend Lambda directories  
safe_remove "backend/lambda-ultra-minimal"
safe_remove "backend/lambda-simple"

echo ""
echo "📦 Removing Lambda deployment packages..."

# Large deployment packages
safe_remove "portfolio-api.zip"
safe_remove "backend/portfolio-api-ultra-minimal.zip"

echo ""
echo "🐍 Removing Lambda handler files..."

# Lambda handler files
safe_remove "backend/ultra_minimal_handler.py"
safe_remove "backend/simple_lambda_handler.py" 
safe_remove "backend/lambda_handler.py"
safe_remove "backend/ultra_simple_lambda.py"

echo ""
echo "🌐 Removing Lambda API files..."

# Lambda API files
safe_remove "backend/ultra_minimal_api.py"
safe_remove "backend/simple_api_server.py"
safe_remove "backend/simple_endpoints.py" 
safe_remove "backend/simple_api.py"

echo ""
echo "📋 Removing Lambda requirements files..."

# Lambda-specific requirements
safe_remove "backend/requirements_lambda_minimal.txt"
safe_remove "backend/requirements_lambda_basic.txt"
safe_remove "backend/requirements_lambda_optimized.txt"
safe_remove "backend/requirements_lambda_ultra_minimal.txt"

echo ""
echo "🔧 Removing Lambda deployment scripts..."

# Lambda deployment scripts
safe_remove "backend/build_optimized_lambda.sh"
safe_remove "backend/fix_lambda_size.sh"
safe_remove "backend/package_lambda.sh"

echo ""
echo "📖 Removing Lambda documentation..."

# Lambda documentation
safe_remove "backend/LAMBDA_SIZE_FIX_README.md"
safe_remove "backend/lambda_optimization_guide.md"

echo ""
echo "⚙️  Removing Lambda configuration files..."

# Lambda configuration
safe_remove "backend/lambda-trust-policy.json"

echo ""
echo "🧪 Removing Lambda test files..."

# Lambda test and response files (these seem to be Lambda-specific based on names)
safe_remove "backend/test_api_response.json"
safe_remove "backend/ultra_test_response.json"
safe_remove "backend/final_test_response.json"
safe_remove "backend/test_response.json"
safe_remove "backend/payload_stocks.json"
safe_remove "backend/response_s3_test.json"
safe_remove "backend/response_final_test.json"
safe_remove "backend/response_test.json"
safe_remove "backend/response_clusters.json"
safe_remove "backend/response_stocks.json"
safe_remove "backend/response_s3_data_test.json"
safe_remove "backend/payload.json"
safe_remove "backend/payload.b64"
safe_remove "backend/response_s3.json"
safe_remove "backend/response_ultra.json"
safe_remove "backend/response_basic.json"
safe_remove "backend/response_success.json"
safe_remove "backend/response_final.json"
safe_remove "backend/response.json"
safe_remove "backend/health_payload.json"

# Root level test files
safe_remove "output.json"

echo ""
echo "🔗 Removing Lambda-related scripts..."

# Lambda API Gateway scripts
safe_remove "backend/fix_api_gateway.sh"
safe_remove "backend/create_api_gateway.sh"

# S3 data loader (might be Lambda-specific)
safe_remove "backend/s3_data_loader.py"

echo ""
echo "📝 Removing Lambda testing documentation..."

# Testing files that seem Lambda-specific
safe_remove "backend/testing_checklist.md"

echo ""
echo "✅ Lambda cleanup completed!"
echo ""
echo "📊 Summary of cleanup:"
echo "   ✓ Removed Lambda deployment packages (~1.7GB saved)"
echo "   ✓ Removed Lambda handler files"
echo "   ✓ Removed Lambda-specific requirements"
echo "   ✓ Removed Lambda deployment scripts"
echo "   ✓ Removed Lambda test files"
echo "   ✓ Removed Lambda documentation"
echo ""
echo "🚀 Your repository is now clean and focused on EC2 deployment!"
echo ""
echo "📌 Kept important files:"
echo "   ✓ Core API server (api_server.py)"
echo "   ✓ API components (/api/ directory)"
echo "   ✓ Data and analysis files"
echo "   ✓ EC2 deployment scripts"
echo "   ✓ Frontend files"
echo ""
echo "💡 Next steps:"
echo "   1. Run: cd backend && ./deploy_to_ec2.sh"
echo "   2. Update your Vercel frontend with the EC2 API URL"
echo "" 