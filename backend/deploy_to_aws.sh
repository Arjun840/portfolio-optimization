#!/bin/bash

# AWS Deployment Script for Portfolio Optimization API
# Usage: ./deploy_to_aws.sh [lambda|ecs|ec2]

set -e

DEPLOYMENT_TYPE=${1:-lambda}
APP_NAME="portfolio-optimization-api"
AWS_REGION=${AWS_REGION:-us-east-1}

echo "🚀 Deploying Portfolio Optimization API to AWS ($DEPLOYMENT_TYPE)"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

case $DEPLOYMENT_TYPE in
    lambda)
        echo "📦 Preparing Lambda deployment..."
        
        # Create deployment directory
        rm -rf lambda-deployment
        mkdir lambda-deployment
        cd lambda-deployment
        
        # Copy application files
        cp -r ../api .
        cp -r ../scripts .
        cp ../api_server.py .
        cp ../lambda_deployment.py .
        cp ../aws_requirements.txt requirements.txt
        
        # Install dependencies
        echo "📥 Installing dependencies..."
        pip install -r requirements.txt -t . --no-deps
        
        # Create deployment package
        echo "📦 Creating deployment package..."
        zip -r ${APP_NAME}.zip . -x "*.pyc" "*/__pycache__/*"
        
        # Check if Lambda function exists
        if aws lambda get-function --function-name $APP_NAME > /dev/null 2>&1; then
            echo "🔄 Updating existing Lambda function..."
            aws lambda update-function-code \
                --function-name $APP_NAME \
                --zip-file fileb://${APP_NAME}.zip
        else
            echo "🆕 Creating new Lambda function..."
            # You'll need to create the IAM role first
            echo "⚠️  Please create an IAM role for Lambda execution first:"
            echo "   aws iam create-role --role-name lambda-execution-role --assume-role-policy-document '{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"lambda.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}'"
            echo "   aws iam attach-role-policy --role-name lambda-execution-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
            
            # Create function (you'll need to replace YOUR_ACCOUNT_ID)
            # aws lambda create-function \
            #     --function-name $APP_NAME \
            #     --runtime python3.9 \
            #     --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
            #     --handler lambda_deployment.handler \
            #     --zip-file fileb://${APP_NAME}.zip \
            #     --timeout 30 \
            #     --memory-size 512
        fi
        
        cd ..
        echo "✅ Lambda deployment complete!"
        echo "📋 Next steps:"
        echo "   1. Set up API Gateway to proxy requests to your Lambda"
        echo "   2. Configure environment variables"
        echo "   3. Set up RDS database"
        ;;
        
    ecs)
        echo "🐳 Preparing ECS deployment..."
        
        # Build Docker image
        echo "🔨 Building Docker image..."
        docker build -t ${APP_NAME}:latest .
        
        # Get AWS account ID
        ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        ECR_URI=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}
        
        # Create ECR repository if it doesn't exist
        aws ecr describe-repositories --repository-names $APP_NAME > /dev/null 2>&1 || \
            aws ecr create-repository --repository-name $APP_NAME
        
        # Login to ECR
        echo "🔐 Logging into ECR..."
        aws ecr get-login-password --region $AWS_REGION | \
            docker login --username AWS --password-stdin $ECR_URI
        
        # Tag and push image
        echo "📤 Pushing image to ECR..."
        docker tag ${APP_NAME}:latest ${ECR_URI}:latest
        docker push ${ECR_URI}:latest
        
        echo "✅ ECS image ready!"
        echo "📋 Next steps:"
        echo "   1. Create ECS cluster"
        echo "   2. Create task definition using image: ${ECR_URI}:latest"
        echo "   3. Create ECS service"
        ;;
        
    ec2)
        echo "🖥️  EC2 deployment instructions:"
        echo "1. Launch EC2 instance (t3.small or larger)"
        echo "2. Install Docker and AWS CLI"
        echo "3. Clone your repository"
        echo "4. Run: docker build -t $APP_NAME ."
        echo "5. Run: docker run -d -p 80:8000 $APP_NAME"
        ;;
        
    *)
        echo "❌ Unknown deployment type: $DEPLOYMENT_TYPE"
        echo "Usage: $0 [lambda|ecs|ec2]"
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment preparation complete!"
echo "📖 See aws_deployment_guide.md for detailed setup instructions" 