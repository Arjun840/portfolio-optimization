#!/bin/bash

# ECS Fargate Deployment Script
# Alternative to Lambda for ML workloads (no size limits)

set -e

echo "🚀 ECS Fargate Deployment for Portfolio Optimization API"
echo "========================================================"

# Configuration
CLUSTER_NAME="portfolio-optimization-cluster"
SERVICE_NAME="portfolio-api-service"
TASK_DEFINITION="portfolio-api-task"
ECR_REPO="portfolio-optimization-api"
REGION="us-east-1"
VPC_CIDR="10.0.0.0/16"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

echo "📍 Account ID: $ACCOUNT_ID"
echo "📍 Region: $REGION"
echo "📦 ECR Repository: $ECR_URI"

# Step 1: Create Dockerfile optimized for production
echo ""
echo "🐳 Step 1: Creating optimized Dockerfile..."
cat > Dockerfile << EOF
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
EOF

echo "✅ Dockerfile created"

# Step 2: Create docker-compose for local testing
echo ""
echo "🔧 Step 2: Creating docker-compose for local testing..."
cat > docker-compose.yml << EOF
version: '3.8'
services:
  portfolio-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
      - DATABASE_URL=sqlite:///./portfolio.db
    volumes:
      - ./data:/app/data:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
EOF

echo "✅ docker-compose.yml created"

# Step 3: Create ECR repository
echo ""
echo "📦 Step 3: Creating ECR repository..."
if aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION >/dev/null 2>&1; then
    echo "⚠️  ECR repository already exists"
else
    aws ecr create-repository --repository-name $ECR_REPO --region $REGION
    echo "✅ ECR repository created"
fi

# Step 4: Build and push Docker image
echo ""
echo "🔨 Step 4: Building and pushing Docker image..."

# Get ECR login token
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

# Build image
echo "Building Docker image..."
docker build -t $ECR_REPO:latest .

# Tag for ECR
docker tag $ECR_REPO:latest $ECR_URI:latest

# Push to ECR
echo "Pushing to ECR..."
docker push $ECR_URI:latest

echo "✅ Docker image pushed to ECR"

# Step 5: Create VPC and networking (if not exists)
echo ""
echo "🌐 Step 5: Setting up VPC and networking..."

# Check if VPC exists
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=tag:Name,Values=portfolio-vpc" --query "Vpcs[0].VpcId" --output text --region $REGION 2>/dev/null)

if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
    echo "Creating VPC..."
    VPC_ID=$(aws ec2 create-vpc --cidr-block $VPC_CIDR --region $REGION --query "Vpc.VpcId" --output text)
    aws ec2 create-tags --resources $VPC_ID --tags Key=Name,Value=portfolio-vpc --region $REGION
    
    # Enable DNS hostnames
    aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames --region $REGION
    
    echo "✅ VPC created: $VPC_ID"
else
    echo "✅ Using existing VPC: $VPC_ID"
fi

# Create Internet Gateway
IGW_ID=$(aws ec2 describe-internet-gateways --filters "Name=tag:Name,Values=portfolio-igw" --query "InternetGateways[0].InternetGatewayId" --output text --region $REGION 2>/dev/null)

if [ "$IGW_ID" = "None" ] || [ -z "$IGW_ID" ]; then
    echo "Creating Internet Gateway..."
    IGW_ID=$(aws ec2 create-internet-gateway --region $REGION --query "InternetGateway.InternetGatewayId" --output text)
    aws ec2 create-tags --resources $IGW_ID --tags Key=Name,Value=portfolio-igw --region $REGION
    aws ec2 attach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID --region $REGION
    echo "✅ Internet Gateway created: $IGW_ID"
else
    echo "✅ Using existing Internet Gateway: $IGW_ID"
fi

# Create public subnets
SUBNET1_ID=$(aws ec2 describe-subnets --filters "Name=tag:Name,Values=portfolio-public-1" --query "Subnets[0].SubnetId" --output text --region $REGION 2>/dev/null)
SUBNET2_ID=$(aws ec2 describe-subnets --filters "Name=tag:Name,Values=portfolio-public-2" --query "Subnets[0].SubnetId" --output text --region $REGION 2>/dev/null)

if [ "$SUBNET1_ID" = "None" ] || [ -z "$SUBNET1_ID" ]; then
    echo "Creating public subnets..."
    SUBNET1_ID=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.1.0/24 --availability-zone ${REGION}a --region $REGION --query "Subnet.SubnetId" --output text)
    aws ec2 create-tags --resources $SUBNET1_ID --tags Key=Name,Value=portfolio-public-1 --region $REGION
    
    SUBNET2_ID=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.2.0/24 --availability-zone ${REGION}b --region $REGION --query "Subnet.SubnetId" --output text)
    aws ec2 create-tags --resources $SUBNET2_ID --tags Key=Name,Value=portfolio-public-2 --region $REGION
    
    echo "✅ Public subnets created: $SUBNET1_ID, $SUBNET2_ID"
else
    echo "✅ Using existing subnets: $SUBNET1_ID, $SUBNET2_ID"
fi

# Create security group
SG_ID=$(aws ec2 describe-security-groups --filters "Name=tag:Name,Values=portfolio-sg" --query "SecurityGroups[0].GroupId" --output text --region $REGION 2>/dev/null)

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group --group-name portfolio-sg --description "Portfolio API Security Group" --vpc-id $VPC_ID --region $REGION --query "GroupId" --output text)
    aws ec2 create-tags --resources $SG_ID --tags Key=Name,Value=portfolio-sg --region $REGION
    
    # Allow HTTP traffic
    aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region $REGION
    
    echo "✅ Security group created: $SG_ID"
else
    echo "✅ Using existing security group: $SG_ID"
fi

# Step 6: Create ECS cluster
echo ""
echo "⚙️  Step 6: Creating ECS cluster..."
if aws ecs describe-clusters --clusters $CLUSTER_NAME --region $REGION >/dev/null 2>&1; then
    echo "✅ ECS cluster already exists"
else
    aws ecs create-cluster --cluster-name $CLUSTER_NAME --capacity-providers FARGATE --region $REGION
    echo "✅ ECS cluster created"
fi

# Step 7: Create task execution role
echo ""
echo "🔐 Step 7: Creating task execution role..."
ROLE_NAME="portfolioEcsTaskExecutionRole"

if aws iam get-role --role-name $ROLE_NAME >/dev/null 2>&1; then
    echo "✅ Task execution role already exists"
else
    # Create trust policy
    cat > task-execution-trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ecs-tasks.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

    aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://task-execution-trust-policy.json
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    
    rm task-execution-trust-policy.json
    echo "✅ Task execution role created"
fi

TASK_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Step 8: Create task definition
echo ""
echo "📋 Step 8: Creating ECS task definition..."
cat > task-definition.json << EOF
{
    "family": "$TASK_DEFINITION",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "$TASK_ROLE_ARN",
    "taskRoleArn": "$TASK_ROLE_ARN",
    "containerDefinitions": [
        {
            "name": "portfolio-api",
            "image": "$ECR_URI:latest",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/portfolio-api",
                    "awslogs-region": "$REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "environment": [
                {
                    "name": "ENV",
                    "value": "production"
                }
            ]
        }
    ]
}
EOF

# Create CloudWatch log group
aws logs create-log-group --log-group-name "/ecs/portfolio-api" --region $REGION 2>/dev/null || echo "Log group already exists"

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json --region $REGION
rm task-definition.json

echo "✅ Task definition registered"

# Step 9: Create ECS service
echo ""
echo "🚀 Step 9: Creating ECS service..."
if aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION >/dev/null 2>&1; then
    echo "⚠️  Service already exists, updating..."
    aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --task-definition $TASK_DEFINITION --region $REGION
else
    cat > service-definition.json << EOF
{
    "serviceName": "$SERVICE_NAME",
    "cluster": "$CLUSTER_NAME",
    "taskDefinition": "$TASK_DEFINITION",
    "desiredCount": 1,
    "launchType": "FARGATE",
    "networkConfiguration": {
        "awsvpcConfiguration": {
            "subnets": ["$SUBNET1_ID", "$SUBNET2_ID"],
            "securityGroups": ["$SG_ID"],
            "assignPublicIp": "ENABLED"
        }
    }
}
EOF

    aws ecs create-service --cli-input-json file://service-definition.json --region $REGION
    rm service-definition.json
    echo "✅ ECS service created"
fi

# Step 10: Get service endpoint
echo ""
echo "🔍 Step 10: Getting service endpoint..."
echo "Waiting for service to be running..."
aws ecs wait services-stable --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION

# Get task ARN
TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $SERVICE_NAME --region $REGION --query "taskArns[0]" --output text)

if [ "$TASK_ARN" != "None" ]; then
    # Get public IP
    PUBLIC_IP=$(aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks $TASK_ARN --region $REGION --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text | xargs -I {} aws ec2 describe-network-interfaces --network-interface-ids {} --region $REGION --query "NetworkInterfaces[0].Association.PublicIp" --output text)
    
    if [ "$PUBLIC_IP" != "None" ]; then
        echo "✅ Service is running!"
        echo "🌐 API Endpoint: http://$PUBLIC_IP:8000"
        echo "📋 Health Check: http://$PUBLIC_IP:8000/health"
        echo "📖 API Docs: http://$PUBLIC_IP:8000/docs"
    fi
fi

echo ""
echo "🎉 ECS DEPLOYMENT COMPLETED!"
echo "============================="
echo ""
echo "📋 Summary:"
echo "- ✅ Docker image built and pushed to ECR"
echo "- ✅ VPC and networking configured"
echo "- ✅ ECS cluster created: $CLUSTER_NAME"
echo "- ✅ ECS service deployed: $SERVICE_NAME"
echo ""
echo "🔧 Management Commands:"
echo "# Scale service:"
echo "aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 2 --region $REGION"
echo ""
echo "# View logs:"
echo "aws logs tail /ecs/portfolio-api --follow --region $REGION"
echo ""
echo "# Redeploy (after code changes):"
echo "docker build -t $ECR_REPO:latest . && docker tag $ECR_REPO:latest $ECR_URI:latest && docker push $ECR_URI:latest"
echo "aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --force-new-deployment --region $REGION"
echo ""
echo "# Stop service:"
echo "aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 0 --region $REGION" 