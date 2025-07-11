#!/bin/bash

# Prerequisites Check Script
# Validates environment before deployment

echo "🔍 Checking Prerequisites for AWS Deployment"
echo "============================================="

ERROR_COUNT=0

# Function to check command availability
check_command() {
    if command -v $1 >/dev/null 2>&1; then
        echo "✅ $1 is installed"
        return 0
    else
        echo "❌ $1 is not installed or not in PATH"
        return 1
    fi
}

# Function to check AWS credentials
check_aws_auth() {
    if aws sts get-caller-identity >/dev/null 2>&1; then
        ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-east-1")
        echo "✅ AWS credentials configured"
        echo "   Account ID: $ACCOUNT_ID"
        echo "   Region: $AWS_REGION"
        return 0
    else
        echo "❌ AWS credentials not configured or invalid"
        echo "   Run: aws configure"
        return 1
    fi
}

# Function to check permissions
check_aws_permissions() {
    echo "🔐 Checking AWS permissions..."
    
    # Check basic permissions
    if aws iam get-user >/dev/null 2>&1 || aws sts get-caller-identity >/dev/null 2>&1; then
        echo "✅ Basic AWS access confirmed"
    else
        echo "❌ Cannot verify AWS access"
        ((ERROR_COUNT++))
    fi
    
    # Check S3 permissions
    if aws s3 ls >/dev/null 2>&1; then
        echo "✅ S3 access confirmed"
    else
        echo "⚠️  S3 access not confirmed (may need permissions)"
    fi
    
    # Check Lambda permissions (for Option 1)
    if aws lambda list-functions --max-items 1 >/dev/null 2>&1; then
        echo "✅ Lambda access confirmed"
    else
        echo "⚠️  Lambda access not confirmed (needed for Option 1)"
    fi
    
    # Check ECS permissions (for Option 2)
    if aws ecs list-clusters --max-items 1 >/dev/null 2>&1; then
        echo "✅ ECS access confirmed"
    else
        echo "⚠️  ECS access not confirmed (needed for Option 2)"
    fi
}

echo ""
echo "🛠️  Checking Required Tools..."

# Check AWS CLI
if ! check_command aws; then
    echo "   Install: https://aws.amazon.com/cli/"
    ((ERROR_COUNT++))
fi

# Check Python
if ! check_command python3; then
    echo "   Install Python 3.8+"
    ((ERROR_COUNT++))
fi

# Check pip
if ! check_command pip; then
    echo "   Install pip"
    ((ERROR_COUNT++))
fi

# Check zip
if ! check_command zip; then
    echo "   Install zip utility"
    ((ERROR_COUNT++))
fi

echo ""
echo "🔐 Checking AWS Configuration..."

# Check AWS credentials
if ! check_aws_auth; then
    echo "   Configure AWS: aws configure"
    echo "   You need: Access Key ID, Secret Access Key, Region"
    ((ERROR_COUNT++))
fi

# Check AWS permissions
check_aws_permissions

echo ""
echo "🐳 Checking Docker (for ECS option)..."

if check_command docker; then
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker is running"
        DOCKER_VERSION=$(docker --version)
        echo "   $DOCKER_VERSION"
    else
        echo "⚠️  Docker is installed but not running"
        echo "   Start Docker Desktop or run: sudo systemctl start docker"
    fi
else
    echo "⚠️  Docker not installed (needed for ECS option)"
    echo "   Install: https://docker.com/get-started"
fi

echo ""
echo "📦 Checking Current Lambda Package..."

if [ -d "lambda-package" ]; then
    PACKAGE_SIZE=$(du -sh lambda-package | cut -f1)
    echo "✅ Lambda package exists: $PACKAGE_SIZE"
    
    if [ -f "portfolio-api.zip" ]; then
        ZIP_SIZE=$(ls -lh portfolio-api.zip | awk '{print $5}')
        echo "✅ Lambda ZIP exists: $ZIP_SIZE"
    else
        echo "⚠️  Lambda ZIP not found"
    fi
else
    echo "❌ Lambda package directory not found"
    echo "   Run the packaging script first"
    ((ERROR_COUNT++))
fi

echo ""
echo "📋 Checking Application Files..."

# Check essential files
REQUIRED_FILES=("api_server.py" "requirements.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file not found"
        ((ERROR_COUNT++))
    fi
done

# Check API directory
if [ -d "api" ]; then
    echo "✅ api/ directory exists"
else
    echo "❌ api/ directory not found"
    ((ERROR_COUNT++))
fi

echo ""
echo "📊 PREREQUISITE CHECK SUMMARY"
echo "=============================="

if [ $ERROR_COUNT -eq 0 ]; then
    echo "🎉 ALL PREREQUISITES MET!"
    echo ""
    echo "✅ You can proceed with either deployment option:"
    echo ""
    echo "   Option 1 (Quick Fix): ./fix_lambda_size.sh"
    echo "   Option 2 (Production): ./deploy_to_ecs.sh"
    echo ""
    echo "🚀 Ready for deployment!"
else
    echo "⚠️  ISSUES FOUND: $ERROR_COUNT"
    echo ""
    echo "❗ Please fix the issues above before proceeding:"
    echo ""
    
    if ! command -v aws >/dev/null 2>&1; then
        echo "1. Install AWS CLI: https://aws.amazon.com/cli/"
    fi
    
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        echo "2. Configure AWS credentials: aws configure"
    fi
    
    if [ ! -d "lambda-package" ]; then
        echo "3. Create Lambda package first"
    fi
    
    echo ""
    echo "Then run this script again: ./check_prerequisites.sh"
fi

echo ""
echo "💡 Need help? Check LAMBDA_SIZE_FIX_README.md" 