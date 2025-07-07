# AWS Deployment Guide for Portfolio Optimization API

## 🚀 Deployment Options Overview

### Option 1: AWS Lambda + API Gateway (Serverless) ⭐ Recommended
- **Cost**: Pay per request (very cost-effective for moderate usage)
- **Scaling**: Automatic scaling from 0 to thousands of concurrent users
- **Maintenance**: Minimal server management required
- **Cold starts**: ~1-3 seconds for first request after inactivity

### Option 2: AWS ECS/Fargate (Containerized)
- **Cost**: Pay for running containers (predictable costs)
- **Scaling**: Container-based auto-scaling
- **Performance**: No cold starts, consistent performance
- **Complexity**: Moderate setup complexity

### Option 3: AWS EC2 (Traditional Server)
- **Cost**: Pay for server instances (24/7 or on-demand)
- **Control**: Full server control and customization
- **Performance**: Consistent, no cold starts
- **Maintenance**: You manage the server

### Option 4: AWS Elastic Beanstalk (Platform-as-a-Service)
- **Ease**: Simplest deployment, just upload your code
- **Cost**: EC2 + Load Balancer costs
- **Management**: AWS handles infrastructure
- **Scaling**: Built-in auto-scaling

## 🏗️ Architecture Components Needed

### 1. Database (Replace In-Memory Storage)
```python
# Current: In-memory user storage
fake_users_db = {...}

# AWS Options:
# - Amazon RDS (PostgreSQL/MySQL) - Managed relational database
# - Amazon DynamoDB - NoSQL, serverless database
# - Amazon DocumentDB - MongoDB-compatible
```

### 2. Data Storage (Portfolio Data Files)
```python
# Current: Local pickle/CSV files
# AWS Options:
# - Amazon S3 - Store data files, highly scalable
# - Amazon EFS - Network file system for containers
# - RDS - Store processed data in database tables
```

### 3. Authentication Options
```python
# Current: Custom JWT (keep this!)
# AWS Enhancement Options:
# - Amazon Cognito - Managed user pools
# - AWS IAM - Service-to-service authentication
# - Keep custom JWT (recommended for your use case)
```

## 📋 Step-by-Step Lambda Deployment

### Step 1: Install AWS CLI and Dependencies
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Install deployment dependencies
pip install mangum boto3
```

### Step 2: Create Lambda Deployment Package
```bash
# Create deployment directory
mkdir lambda-deployment
cd lambda-deployment

# Copy your application files
cp -r ../api/ .
cp ../api_server.py .
cp ../lambda_deployment.py .

# Install dependencies in deployment directory
pip install -r ../aws_requirements.txt -t .

# Create deployment package
zip -r portfolio-api.zip .
```

### Step 3: Create Lambda Function
```bash
# Create Lambda function
aws lambda create-function \
  --function-name portfolio-optimization-api \
  --runtime python3.9 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
  --handler lambda_deployment.handler \
  --zip-file fileb://portfolio-api.zip
```

### Step 4: Create API Gateway
```bash
# Create REST API
aws apigateway create-rest-api --name portfolio-api

# Set up proxy integration (detailed steps in AWS console)
```

## 🔧 Required AWS Services Setup

### 1. Amazon RDS for User Database
```sql
-- Create users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Amazon S3 for Data Storage
```python
# Modified data loading for S3
import boto3

def load_data_from_s3():
    s3 = boto3.client('s3')
    
    # Download data files from S3
    s3.download_file('your-portfolio-bucket', 'data/cleaned_prices.pkl', '/tmp/cleaned_prices.pkl')
    s3.download_file('your-portfolio-bucket', 'data/cleaned_returns.pkl', '/tmp/cleaned_returns.pkl')
    
    # Load as usual
    prices = pd.read_pickle('/tmp/cleaned_prices.pkl')
    returns = pd.read_pickle('/tmp/cleaned_returns.pkl')
    return prices, returns
```

### 3. Environment Variables Setup
```bash
# Set Lambda environment variables
aws lambda update-function-configuration \
  --function-name portfolio-optimization-api \
  --environment Variables='{
    "JWT_SECRET_KEY":"your-production-secret-key",
    "DATABASE_URL":"postgresql://user:pass@rds-endpoint:5432/portfolio",
    "S3_BUCKET":"your-portfolio-data-bucket",
    "AWS_REGION":"us-east-1"
  }'
```

## 🗄️ Database Integration Example

### Modified auth.py for RDS
```python
import os
import psycopg2
from contextlib import contextmanager

DATABASE_URL = os.environ.get('DATABASE_URL')

@contextmanager
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

def register_user(email: str, password: str, full_name: str) -> dict:
    hashed_password = get_password_hash(password)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (email, hashed_password, full_name) VALUES (%s, %s, %s) RETURNING id",
                (email, hashed_password, full_name)
            )
            user_id = cursor.fetchone()[0]
            conn.commit()
            
            return {
                "id": user_id,
                "email": email,
                "full_name": full_name
            }
        except psycopg2.IntegrityError:
            raise HTTPException(status_code=400, detail="Email already registered")
```

## 💰 Cost Estimates (Monthly)

### Lambda + API Gateway (Serverless)
- **Light usage** (1K requests/month): ~$1-5
- **Moderate usage** (10K requests/month): ~$5-15
- **Heavy usage** (100K requests/month): ~$15-50

### ECS Fargate
- **Single container**: ~$15-30/month
- **Auto-scaling**: ~$30-100/month

### EC2
- **t3.small**: ~$15/month
- **t3.medium**: ~$30/month
- **Load balancer**: +$20/month

### Database Costs
- **RDS t3.micro**: ~$15/month
- **DynamoDB**: Pay per request (~$1-10/month for typical usage)

## 🔒 Security Considerations

### 1. API Gateway Settings
```json
{
  "throttle": {
    "rateLimit": 1000,
    "burstLimit": 2000
  },
  "cors": {
    "allowOrigins": ["https://your-frontend-domain.com"],
    "allowMethods": ["GET", "POST", "OPTIONS"]
  }
}
```

### 2. Lambda Security
- Use IAM roles with minimal permissions
- Enable CloudWatch logging
- Set up VPC if needed for database access

### 3. Environment Variables
- Store secrets in AWS Systems Manager Parameter Store
- Use AWS Secrets Manager for database credentials

## 📊 Monitoring & Logging

### CloudWatch Integration
```python
import logging
import json

# Configure CloudWatch logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info(f"Request received: {json.dumps(event)}")
    
    try:
        result = handler(event, context)
        logger.info("Request processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise
```

## 🚀 Quick Start Commands

### Deploy to Lambda (Simplified)
```bash
# 1. Package application
./scripts/package_for_lambda.sh

# 2. Deploy using AWS SAM (recommended)
sam init --runtime python3.9
sam build
sam deploy --guided

# 3. Or use Serverless Framework
serverless deploy
```

### Deploy to ECS
```bash
# 1. Build Docker image
docker build -t portfolio-api .

# 2. Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin ECR_URI
docker tag portfolio-api:latest ECR_URI/portfolio-api:latest
docker push ECR_URI/portfolio-api:latest

# 3. Deploy to ECS
aws ecs create-service --cluster portfolio-cluster --service-name portfolio-api
```

## 🎯 Recommended Next Steps

1. **Start with Lambda** - Easiest and most cost-effective
2. **Set up RDS database** - Replace in-memory user storage
3. **Move data to S3** - Store portfolio data files
4. **Add monitoring** - CloudWatch logs and metrics
5. **Set up CI/CD** - GitHub Actions or AWS CodePipeline

Your current API is already well-structured for AWS deployment! The main changes needed are:
- Database integration (replace in-memory storage)
- Data file storage (S3 instead of local files)
- Environment configuration
- Lambda adapter (already created above) 