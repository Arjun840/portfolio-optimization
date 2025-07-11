# 🚀 Complete AWS Deployment Guide - Portfolio Optimization Platform

A comprehensive guide to deploy your full-stack portfolio optimization application on AWS with production-ready architecture, security, and CI/CD.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Backend Deployment](#backend-deployment)
- [Frontend Deployment](#frontend-deployment)
- [Database Setup](#database-setup)
- [Domain & SSL](#domain--ssl)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Logging](#monitoring--logging)
- [Security Best Practices](#security-best-practices)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture Overview

### Recommended Production Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CloudFront    │────│   S3 + Amplify   │    │   API Gateway   │
│   (Global CDN)  │    │   (Frontend)     │────│   (Rate Limit)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│      Route53    │    │   Lambda/ECS     │    │   ElastiCache   │
│   (DNS + SSL)   │    │   (Backend API)  │────│   (Caching)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                       ┌─────────────────┐    ┌─────────────────┐
                       │       RDS       │    │       S3        │
                       │   (Database)    │    │  (Data Files)   │
                       └─────────────────┘    └─────────────────┘
```

### Services Used
- **Frontend**: AWS Amplify + S3 + CloudFront
- **Backend**: Lambda + API Gateway (or ECS Fargate)
- **Database**: RDS PostgreSQL
- **File Storage**: S3
- **DNS**: Route53
- **SSL**: Certificate Manager
- **Monitoring**: CloudWatch
- **CI/CD**: GitHub Actions + AWS CodePipeline

## 🔧 Prerequisites

### 1. AWS Account Setup
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, region (us-east-1), and output format (json)

# Verify installation
aws sts get-caller-identity
```

### 2. Install Required Tools
```bash
# Node.js and npm (for frontend)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# AWS CDK (optional, for infrastructure as code)
npm install -g aws-cdk

# Amplify CLI (for frontend deployment)
npm install -g @aws-amplify/cli

# Docker (for containerized deployment)
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER
```

### 3. Domain Setup
- Purchase domain through Route53 or transfer existing domain
- Create hosted zone in Route53
- Update nameservers at your domain registrar

## 🖥️ Backend Deployment

### Option A: Serverless (Lambda + API Gateway) ⭐ Recommended

#### Step 1: Prepare Lambda Deployment Package
```bash
# Navigate to backend directory
cd backend

# Create deployment package
mkdir lambda-package
cd lambda-package

# Copy application files
cp -r ../api/ .
cp ../api_server.py .
cp -r ../scripts/ .
cp -r ../data/ .

# Create Lambda handler
cat > lambda_handler.py << 'EOF'
import json
from mangum import Mangum
from api_server import app

# Create the Lambda handler
handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        response = handler(event, context)
        return response
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
EOF

# Install dependencies
pip install mangum boto3 psycopg2-binary -t .
pip install -r ../requirements.txt -t .

# Create deployment zip
zip -r ../portfolio-api.zip . -x "*.pyc" "*/__pycache__/*"
cd ..
```

#### Step 2: Create IAM Role for Lambda
```bash
# Create trust policy
cat > lambda-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create IAM role
aws iam create-role \
  --role-name PortfolioLambdaRole \
  --assume-role-policy-document file://lambda-trust-policy.json

# Attach basic execution policy
aws iam attach-role-policy \
  --role-name PortfolioLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Attach S3 access policy
aws iam attach-role-policy \
  --role-name PortfolioLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

#### Step 3: Deploy Lambda Function
```bash
# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create Lambda function
aws lambda create-function \
  --function-name portfolio-optimization-api \
  --runtime python3.9 \
  --role arn:aws:iam::${ACCOUNT_ID}:role/PortfolioLambdaRole \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://portfolio-api.zip \
  --timeout 30 \
  --memory-size 512

# Set environment variables
aws lambda update-function-configuration \
  --function-name portfolio-optimization-api \
  --environment Variables='{
    "JWT_SECRET_KEY":"your-super-secret-key-change-this-in-production",
    "DATABASE_URL":"postgresql://username:password@your-rds-endpoint:5432/portfolio",
    "S3_BUCKET":"your-portfolio-data-bucket",
    "ENVIRONMENT":"production"
  }'
```

#### Step 4: Create API Gateway
```bash
# Create REST API
API_ID=$(aws apigateway create-rest-api \
  --name portfolio-optimization-api \
  --query 'id' --output text)

# Get root resource ID
ROOT_ID=$(aws apigateway get-resources \
  --rest-api-id $API_ID \
  --query 'items[0].id' --output text)

# Create proxy resource
RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ROOT_ID \
  --path-part '{proxy+}' \
  --query 'id' --output text)

# Create ANY method
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $RESOURCE_ID \
  --http-method ANY \
  --authorization-type NONE

# Set up Lambda integration
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $RESOURCE_ID \
  --http-method ANY \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:${ACCOUNT_ID}:function:portfolio-optimization-api/invocations

# Give API Gateway permission to invoke Lambda
aws lambda add-permission \
  --function-name portfolio-optimization-api \
  --statement-id allow-api-gateway \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:us-east-1:${ACCOUNT_ID}:${API_ID}/*/*"

# Deploy API
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name prod

echo "API URL: https://${API_ID}.execute-api.us-east-1.amazonaws.com/prod"
```

### Option B: Containerized (ECS Fargate)

#### Step 1: Create Dockerfile
```dockerfile
# Create Dockerfile in backend directory
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 2: Build and Push to ECR
```bash
# Create ECR repository
aws ecr create-repository --repository-name portfolio-api

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Build and push image
docker build -t portfolio-api .
docker tag portfolio-api:latest ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/portfolio-api:latest
docker push ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/portfolio-api:latest
```

#### Step 3: Create ECS Cluster and Service
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name portfolio-cluster

# Create task definition (save as task-definition.json)
cat > task-definition.json << 'EOF'
{
  "family": "portfolio-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "portfolio-api",
      "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/portfolio-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/portfolio-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "environment": [
        {
          "name": "JWT_SECRET_KEY",
          "value": "your-super-secret-key"
        },
        {
          "name": "DATABASE_URL",
          "value": "postgresql://username:password@your-rds-endpoint:5432/portfolio"
        }
      ]
    }
  ]
}
EOF

# Replace ACCOUNT_ID with actual account ID
sed -i "s/ACCOUNT_ID/${ACCOUNT_ID}/g" task-definition.json

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

## 🌐 Frontend Deployment

### Option A: AWS Amplify (Recommended)

#### Step 1: Initialize Amplify
```bash
# Navigate to frontend directory
cd frontend/frontend

# Initialize Amplify
amplify init

# Follow prompts:
# Project name: portfolio-optimization
# Environment name: prod
# Default editor: Visual Studio Code
# App type: javascript
# Framework: react
# Source directory: src
# Distribution directory: dist
# Build command: npm run build
# Start command: npm run dev
```

#### Step 2: Add Hosting
```bash
# Add hosting with CloudFront and S3
amplify add hosting

# Choose:
# Select the plugin module: Hosting with CloudFront + S3
# Select the environment setup: PROD (S3 with CloudFront using HTTPS)
# hosting bucket name: (default)
# index doc for the website: index.html
# error doc for the website: index.html

# Deploy
amplify publish
```

#### Step 3: Configure Environment Variables
```bash
# Create .env.production file
cat > .env.production << 'EOF'
VITE_API_URL=https://your-api-id.execute-api.us-east-1.amazonaws.com/prod
VITE_APP_NAME=PortfolioMax
VITE_ENVIRONMENT=production
EOF

# Update build settings in Amplify Console:
# Build settings > Environment variables
# Add VITE_API_URL with your API Gateway URL
```

### Option B: Manual S3 + CloudFront

#### Step 1: Create S3 Bucket
```bash
# Create S3 bucket for website
aws s3 mb s3://portfolio-optimization-frontend --region us-east-1

# Enable static website hosting
aws s3 website s3://portfolio-optimization-frontend \
  --index-document index.html \
  --error-document index.html

# Upload website files (after building)
cd frontend/frontend
npm run build
aws s3 sync dist/ s3://portfolio-optimization-frontend --delete
```

#### Step 2: Create CloudFront Distribution
```bash
# Create distribution configuration
cat > cloudfront-config.json << 'EOF'
{
  "CallerReference": "portfolio-frontend-$(date +%s)",
  "Comment": "Portfolio Optimization Frontend",
  "DefaultRootObject": "index.html",
  "Origins": {
    "Quantity": 1,
    "Items": [
      {
        "Id": "S3-portfolio-optimization-frontend",
        "DomainName": "portfolio-optimization-frontend.s3.amazonaws.com",
        "S3OriginConfig": {
          "OriginAccessIdentity": ""
        }
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "S3-portfolio-optimization-frontend",
    "ViewerProtocolPolicy": "redirect-to-https",
    "MinTTL": 0,
    "ForwardedValues": {
      "QueryString": false,
      "Cookies": {"Forward": "none"}
    }
  },
  "Enabled": true,
  "PriceClass": "PriceClass_100"
}
EOF

# Create CloudFront distribution
aws cloudfront create-distribution --distribution-config file://cloudfront-config.json
```

## 🗄️ Database Setup

### Step 1: Create RDS PostgreSQL Instance
```bash
# Create DB subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name portfolio-db-subnet \
  --db-subnet-group-description "Portfolio DB Subnet Group" \
  --subnet-ids subnet-12345678 subnet-87654321

# Create security group for RDS
aws ec2 create-security-group \
  --group-name portfolio-db-sg \
  --description "Portfolio Database Security Group"

# Allow inbound on port 5432 from Lambda/ECS
aws ec2 authorize-security-group-ingress \
  --group-name portfolio-db-sg \
  --protocol tcp \
  --port 5432 \
  --cidr 0.0.0.0/0

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier portfolio-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username dbadmin \
  --master-user-password YourSecurePassword123! \
  --allocated-storage 20 \
  --vpc-security-group-ids sg-your-security-group-id \
  --db-subnet-group-name portfolio-db-subnet \
  --backup-retention-period 7 \
  --multi-az \
  --storage-encrypted
```

### Step 2: Initialize Database Schema
```sql
-- Connect to your RDS instance and run:

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Saved portfolios table
CREATE TABLE saved_portfolios (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    portfolio_name VARCHAR(255) NOT NULL,
    description TEXT,
    portfolio_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_saved_portfolios_user_id ON saved_portfolios(user_id);
CREATE INDEX idx_saved_portfolios_created_at ON saved_portfolios(created_at);
```

### Step 3: Upload Data Files to S3
```bash
# Create S3 bucket for data
aws s3 mb s3://portfolio-optimization-data

# Upload data files
aws s3 sync backend/data/ s3://portfolio-optimization-data/data/ \
  --exclude "*.pyc" --exclude "__pycache__/*"

# Set appropriate permissions
aws s3api put-bucket-policy --bucket portfolio-optimization-data --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowLambdaAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::ACCOUNT_ID:role/PortfolioLambdaRole"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::portfolio-optimization-data/*"
    }
  ]
}'
```

## 🌍 Domain & SSL Setup

### Step 1: Request SSL Certificate
```bash
# Request certificate through ACM
aws acm request-certificate \
  --domain-name yourdomain.com \
  --subject-alternative-names www.yourdomain.com api.yourdomain.com \
  --validation-method DNS

# Note the CertificateArn from the response
```

### Step 2: Configure Route53
```bash
# Create hosted zone (if not exists)
aws route53 create-hosted-zone \
  --name yourdomain.com \
  --caller-reference $(date +%s)

# Create A record for frontend (CloudFront)
aws route53 change-resource-record-sets \
  --hosted-zone-id YOUR_HOSTED_ZONE_ID \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "yourdomain.com",
        "Type": "A",
        "AliasTarget": {
          "DNSName": "your-cloudfront-domain.cloudfront.net",
          "EvaluateTargetHealth": false,
          "HostedZoneId": "Z2FDTNDATAQYW2"
        }
      }
    }]
  }'

# Create A record for API
aws route53 change-resource-record-sets \
  --hosted-zone-id YOUR_HOSTED_ZONE_ID \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.yourdomain.com",
        "Type": "A",
        "AliasTarget": {
          "DNSName": "your-api-gateway-domain",
          "EvaluateTargetHealth": false,
          "HostedZoneId": "ZLY8HYME6SFDD"
        }
      }
    }]
  }'
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# Create .github/workflows/deploy.yml
name: Deploy to AWS

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-1

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Package Lambda function
      run: |
        cd backend
        mkdir lambda-package
        cp -r api/ scripts/ data/ lambda-package/
        cp api_server.py lambda-package/
        cd lambda-package
        pip install mangum boto3 psycopg2-binary -t .
        pip install -r ../requirements.txt -t .
        zip -r ../portfolio-api.zip . -x "*.pyc" "*/__pycache__/*"
    
    - name: Update Lambda function
      run: |
        aws lambda update-function-code \
          --function-name portfolio-optimization-api \
          --zip-file fileb://backend/portfolio-api.zip

  deploy-frontend:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    needs: deploy-backend
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: frontend/frontend/package-lock.json
    
    - name: Install dependencies
      run: |
        cd frontend/frontend
        npm ci
    
    - name: Build application
      run: |
        cd frontend/frontend
        npm run build
      env:
        VITE_API_URL: ${{ secrets.VITE_API_URL }}
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Deploy to S3
      run: |
        cd frontend/frontend
        aws s3 sync dist/ s3://portfolio-optimization-frontend --delete
    
    - name: Invalidate CloudFront
      run: |
        aws cloudfront create-invalidation \
          --distribution-id ${{ secrets.CLOUDFRONT_DISTRIBUTION_ID }} \
          --paths "/*"
```

## 📊 Monitoring & Logging

### CloudWatch Setup
```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
  --dashboard-name "Portfolio-Optimization" \
  --dashboard-body '{
    "widgets": [
      {
        "type": "metric",
        "properties": {
          "metrics": [
            ["AWS/Lambda", "Duration", "FunctionName", "portfolio-optimization-api"],
            ["AWS/Lambda", "Errors", "FunctionName", "portfolio-optimization-api"],
            ["AWS/Lambda", "Invocations", "FunctionName", "portfolio-optimization-api"]
          ],
          "period": 300,
          "stat": "Average",
          "region": "us-east-1",
          "title": "Lambda Metrics"
        }
      }
    ]
  }'

# Create CloudWatch alarms
aws cloudwatch put-metric-alarm \
  --alarm-name "Portfolio-API-Errors" \
  --alarm-description "Alert on Lambda errors" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=portfolio-optimization-api \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:portfolio-alerts
```

### Application Logging
```python
# Update backend/api/endpoints.py to include structured logging
import logging
import json
from datetime import datetime

# Configure CloudWatch logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log_request(request_info: dict):
    """Log request information for monitoring"""
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": "INFO",
        "message": "API Request",
        "data": request_info
    }
    logger.info(json.dumps(log_data))

# Example usage in endpoints
async def optimize_portfolio(request: OptimizationRequest, current_user: dict = Depends(get_current_user)):
    log_request({
        "endpoint": "optimize_portfolio",
        "user": current_user.get("email"),
        "parameters": request.dict()
    })
    # ... rest of function
```

## 🔒 Security Best Practices

### 1. API Security
```python
# Add rate limiting and security headers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### 2. Environment Secrets
```bash
# Store secrets in AWS Systems Manager
aws ssm put-parameter \
  --name "/portfolio/prod/jwt-secret" \
  --value "your-super-secret-jwt-key" \
  --type "SecureString"

aws ssm put-parameter \
  --name "/portfolio/prod/database-url" \
  --value "postgresql://user:pass@endpoint:5432/db" \
  --type "SecureString"

# Update Lambda to use SSM parameters
aws lambda update-function-configuration \
  --function-name portfolio-optimization-api \
  --environment Variables='{
    "JWT_SECRET_PARAM":"/portfolio/prod/jwt-secret",
    "DATABASE_URL_PARAM":"/portfolio/prod/database-url"
  }'
```

### 3. Network Security
```bash
# Create VPC for backend resources
aws ec2 create-vpc --cidr-block 10.0.0.0/16

# Create private subnets for RDS
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.1.0/24
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.2.0/24

# Security groups with minimal access
aws ec2 create-security-group \
  --group-name portfolio-rds-sg \
  --description "RDS Security Group" \
  --vpc-id vpc-12345678

# Only allow access from Lambda security group
aws ec2 authorize-security-group-ingress \
  --group-id sg-rds-id \
  --protocol tcp \
  --port 5432 \
  --source-group sg-lambda-id
```

## 💰 Cost Optimization

### Monthly Cost Estimates

#### Serverless Architecture (Recommended)
```
Component                    Light Use    Moderate Use    Heavy Use
─────────────────────────────────────────────────────────────────
Lambda (1M requests/month)   $0.20        $2.00          $20.00
API Gateway                  $3.50        $35.00         $350.00
RDS t3.micro                 $12.50       $12.50         $12.50
S3 Storage (10GB)            $0.23        $0.23          $0.23
CloudFront (1TB transfer)    $8.50        $8.50          $8.50
Route53                      $0.50        $0.50          $0.50
─────────────────────────────────────────────────────────────────
TOTAL                        ~$25/month   ~$60/month     ~$390/month

Light Use: 1K API calls/month, 1K users
Moderate Use: 10K API calls/month, 10K users  
Heavy Use: 100K API calls/month, 100K users
```

#### Cost Optimization Tips
```bash
# 1. Use Reserved Instances for RDS (save 30-60%)
aws rds purchase-reserved-db-instances-offering \
  --reserved-db-instances-offering-id offering-id

# 2. Enable S3 Intelligent Tiering
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket portfolio-optimization-data \
  --id EntireBucket \
  --intelligent-tiering-configuration '{
    "Id": "EntireBucket",
    "Status": "Enabled",
    "Filter": {},
    "Tiering": {
      "Days": 1,
      "AccessTier": "ARCHIVE_ACCESS"
    }
  }'

# 3. Set up CloudWatch billing alerts
aws cloudwatch put-metric-alarm \
  --alarm-name "Portfolio-Billing-Alert" \
  --alarm-description "Alert when monthly bill exceeds $100" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=Currency,Value=USD
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Lambda Cold Start Issues
```python
# Optimize imports and initialization
import json
import os
from functools import lru_cache

# Cache database connections
@lru_cache(maxsize=1)
def get_db_connection():
    return create_connection(os.environ['DATABASE_URL'])

# Pre-load data if possible
@lru_cache(maxsize=1)
def load_portfolio_data():
    return load_data_from_s3()
```

#### 2. CORS Issues
```javascript
// Frontend: Update API calls to include credentials
const response = await fetch(`${API_URL}/optimize`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  credentials: 'include',
  body: JSON.stringify(data)
});
```

#### 3. Database Connection Issues
```python
# Implement connection pooling
import psycopg2.pool

# Create connection pool
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    host=db_host,
    database=db_name,
    user=db_user,
    password=db_password
)

def get_db_connection():
    return connection_pool.getconn()

def return_db_connection(conn):
    connection_pool.putconn(conn)
```

#### 4. Performance Issues
```bash
# Enable CloudWatch X-Ray tracing
aws lambda update-function-configuration \
  --function-name portfolio-optimization-api \
  --tracing-config Mode=Active

# Add CloudFront caching for static content
aws cloudfront update-distribution \
  --id YOUR_DISTRIBUTION_ID \
  --distribution-config '{
    "CacheBehaviors": {
      "Items": [{
        "PathPattern": "/static/*",
        "TTL": 86400
      }]
    }
  }'
```

### Monitoring Commands
```bash
# Check Lambda function logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/portfolio"

# Monitor API Gateway metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApiGateway \
  --metric-name Count \
  --dimensions Name=ApiName,Value=portfolio-optimization-api \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Sum

# Check RDS performance
aws rds describe-db-instances \
  --db-instance-identifier portfolio-db \
  --query 'DBInstances[0].DBInstanceStatus'
```

## 🚀 Quick Start Deployment Script

Save this as `deploy.sh` for one-command deployment:

```bash
#!/bin/bash

# Portfolio Optimization AWS Deployment Script
set -e

echo "🚀 Starting AWS deployment..."

# Variables
PROJECT_NAME="portfolio-optimization"
AWS_REGION="us-east-1"
DOMAIN_NAME="yourdomain.com"  # Change this!

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "📝 AWS Account ID: $ACCOUNT_ID"

# 1. Deploy Backend (Lambda)
echo "🔨 Deploying backend..."
cd backend
./package_lambda.sh
aws lambda update-function-code \
  --function-name portfolio-optimization-api \
  --zip-file fileb://portfolio-api.zip

# 2. Deploy Frontend (S3 + CloudFront)
echo "🌐 Deploying frontend..."
cd ../frontend/frontend
npm install
npm run build
aws s3 sync dist/ s3://portfolio-optimization-frontend --delete

# 3. Invalidate CloudFront cache
DISTRIBUTION_ID=$(aws cloudfront list-distributions \
  --query "DistributionList.Items[?Comment=='Portfolio Optimization Frontend'].Id" \
  --output text)

if [ ! -z "$DISTRIBUTION_ID" ]; then
  aws cloudfront create-invalidation \
    --distribution-id $DISTRIBUTION_ID \
    --paths "/*"
  echo "✅ CloudFront cache invalidated"
fi

echo "🎉 Deployment complete!"
echo "Frontend: https://$DOMAIN_NAME"
echo "API: https://api.$DOMAIN_NAME"
```

## 📞 Support and Next Steps

### Recommended Deployment Order:
1. **Start Simple**: Deploy backend with Lambda + API Gateway
2. **Add Frontend**: Use Amplify for quick frontend deployment  
3. **Database**: Set up RDS and migrate from in-memory storage
4. **Domain**: Configure custom domain with SSL
5. **Monitoring**: Add CloudWatch dashboards and alerts
6. **CI/CD**: Implement automated deployments
7. **Optimization**: Add caching, performance monitoring

### Additional Resources:
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [FastAPI on AWS Lambda](https://mangum.io/)
- [React on AWS Amplify](https://docs.amplify.aws/)
- [AWS Cost Calculator](https://calculator.aws/)

### Need Help?
- AWS Support: Business/Enterprise plans include technical support
- Community: AWS re:Post, Stack Overflow
- Documentation: AWS official docs

Your portfolio optimization platform is now production-ready! 🚀 