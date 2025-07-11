#!/bin/bash

# Portfolio Backend EC2 Deployment Script
# This script automates the deployment of your FastAPI backend to EC2

set -e  # Exit on any error

echo "🚀 Starting Portfolio Backend EC2 Deployment..."

# Configuration
KEY_NAME="portfolio-backend-key"
SECURITY_GROUP_NAME="portfolio-backend-sg"
INSTANCE_TYPE="t3.medium"
INSTANCE_NAME="portfolio-backend"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS CLI is configured"

# Function to create key pair if it doesn't exist
create_key_pair() {
    if [ ! -f "${KEY_NAME}.pem" ]; then
        echo "🔑 Creating key pair..."
        aws ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text > ${KEY_NAME}.pem
        chmod 400 ${KEY_NAME}.pem
        echo "✅ Key pair created: ${KEY_NAME}.pem"
    else
        echo "✅ Key pair already exists: ${KEY_NAME}.pem"
    fi
}

# Function to create security group
create_security_group() {
    # Check if security group exists
    if aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME > /dev/null 2>&1; then
        echo "✅ Security group already exists: $SECURITY_GROUP_NAME"
        SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
            --group-names $SECURITY_GROUP_NAME \
            --query 'SecurityGroups[0].GroupId' \
            --output text)
    else
        echo "🛡️  Creating security group..."
        aws ec2 create-security-group \
            --group-name $SECURITY_GROUP_NAME \
            --description "Security group for portfolio optimization backend"
        
        SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
            --group-names $SECURITY_GROUP_NAME \
            --query 'SecurityGroups[0].GroupId' \
            --output text)
        
        echo "🔓 Configuring security group rules..."
        # SSH access
        aws ec2 authorize-security-group-ingress \
            --group-id $SECURITY_GROUP_ID \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0

        # HTTP access
        aws ec2 authorize-security-group-ingress \
            --group-id $SECURITY_GROUP_ID \
            --protocol tcp \
            --port 80 \
            --cidr 0.0.0.0/0

        # HTTPS access
        aws ec2 authorize-security-group-ingress \
            --group-id $SECURITY_GROUP_ID \
            --protocol tcp \
            --port 443 \
            --cidr 0.0.0.0/0

        # FastAPI access (for testing)
        aws ec2 authorize-security-group-ingress \
            --group-id $SECURITY_GROUP_ID \
            --protocol tcp \
            --port 8000 \
            --cidr 0.0.0.0/0
        
        echo "✅ Security group configured: $SECURITY_GROUP_ID"
    fi
}

# Function to launch EC2 instance
launch_instance() {
    # Check if instance already exists
    EXISTING_INSTANCE=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$EXISTING_INSTANCE" != "None" ] && [ "$EXISTING_INSTANCE" != "null" ]; then
        echo "✅ Instance already exists: $EXISTING_INSTANCE"
        INSTANCE_ID=$EXISTING_INSTANCE
    else
        echo "🚀 Launching EC2 instance..."
        # Get the latest Ubuntu AMI ID
        AMI_ID=$(aws ec2 describe-images \
            --owners 099720109477 \
            --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
            --output text)
        
        echo "📀 Using AMI: $AMI_ID"
        
        INSTANCE_ID=$(aws ec2 run-instances \
            --image-id $AMI_ID \
            --count 1 \
            --instance-type $INSTANCE_TYPE \
            --key-name $KEY_NAME \
            --security-group-ids $SECURITY_GROUP_ID \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
            --query 'Instances[0].InstanceId' \
            --output text)
        
        echo "⏳ Waiting for instance to be running..."
        aws ec2 wait instance-running --instance-ids $INSTANCE_ID
        echo "✅ Instance is running: $INSTANCE_ID"
    fi
}

# Function to get public IP
get_public_ip() {
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo "🌐 Public IP: $PUBLIC_IP"
}

# Function to create deployment package
create_deployment_package() {
    echo "📦 Creating deployment package..."
    tar -czf portfolio-backend.tar.gz \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.log' \
        --exclude='analysis_plots' \
        --exclude='lambda-*' \
        --exclude='*.zip' \
        .
    echo "✅ Deployment package created: portfolio-backend.tar.gz"
}

# Function to upload and setup on EC2
setup_on_ec2() {
    echo "📤 Uploading code to EC2..."
    
    # Wait for SSH to be available
    echo "⏳ Waiting for SSH to be available..."
    while ! ssh -i ${KEY_NAME}.pem -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP echo "SSH Ready" > /dev/null 2>&1; do
        sleep 10
        echo "   Still waiting for SSH..."
    done
    echo "✅ SSH is ready"
    
    # Upload deployment package
    scp -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no portfolio-backend.tar.gz ubuntu@$PUBLIC_IP:/tmp/
    
    # Run setup commands on EC2
    ssh -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'EOF'
        set -e
        echo "🔧 Setting up server environment..."
        
        # Update system
        sudo apt update && sudo apt upgrade -y
        
        # Install required packages
        sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip nginx git curl
        
        # Create application directory
        sudo mkdir -p /opt/portfolio-backend
        sudo chown ubuntu:ubuntu /opt/portfolio-backend
        
        # Extract application code
        cd /opt/portfolio-backend
        tar -xzf /tmp/portfolio-backend.tar.gz
        rm /tmp/portfolio-backend.tar.gz
        
        # Create virtual environment
        python3.11 -m venv venv
        source venv/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip
        
        # Install core dependencies
        pip install fastapi uvicorn[standard] python-multipart
        pip install pandas numpy scipy scikit-learn
        pip install yfinance requests boto3
        pip install python-jose[cryptography] passlib[bcrypt]
        pip install matplotlib seaborn plotly kaleido
        
        # Install additional requirements if available
        if [ -f requirements.txt ]; then
            pip install -r requirements.txt
        fi
        
        echo "✅ Python environment set up successfully"
EOF
    
    echo "✅ Code uploaded and environment configured"
}

# Function to configure services
configure_services() {
    echo "⚙️  Configuring services..."
    
    # Prompt for Vercel URL
    echo "Please enter your Vercel frontend URL (e.g., https://your-app.vercel.app):"
    read -r FRONTEND_URL
    
    ssh -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << EOF
        set -e
        cd /opt/portfolio-backend
        
        # Create environment file
        cat > .env << EOL
ENVIRONMENT=production
FRONTEND_URL=$FRONTEND_URL
SECRET_KEY=\$(openssl rand -hex 32)
JWT_SECRET_KEY=\$(openssl rand -hex 32)
AWS_REGION=\$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
EOL
        
        # Configure Nginx
        sudo tee /etc/nginx/sites-available/portfolio-backend << 'NGINXCONF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
        add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range' always;
        
        if (\$request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS';
            add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization';
            add_header 'Access-Control-Max-Age' 1728000;
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }
    }
}
NGINXCONF
        
        # Enable Nginx site
        sudo ln -sf /etc/nginx/sites-available/portfolio-backend /etc/nginx/sites-enabled/
        sudo rm -f /etc/nginx/sites-enabled/default
        sudo nginx -t
        sudo systemctl restart nginx
        sudo systemctl enable nginx
        
        # Create systemd service
        sudo tee /etc/systemd/system/portfolio-backend.service << 'SERVICECONF'
[Unit]
Description=Portfolio Optimization Backend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/portfolio-backend
Environment=PATH=/opt/portfolio-backend/venv/bin
EnvironmentFile=/opt/portfolio-backend/.env
ExecStart=/opt/portfolio-backend/venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
SERVICECONF
        
        # Start services
        sudo systemctl daemon-reload
        sudo systemctl start portfolio-backend
        sudo systemctl enable portfolio-backend
        
        echo "✅ Services configured and started"
EOF
    
    echo "✅ Services configured successfully"
}

# Function to test deployment
test_deployment() {
    echo "🧪 Testing deployment..."
    
    # Test API health
    if curl -f -s "http://$PUBLIC_IP/" > /dev/null; then
        echo "✅ API is responding at http://$PUBLIC_IP/"
        echo "📖 API documentation available at: http://$PUBLIC_IP/docs"
    else
        echo "❌ API is not responding. Check the logs:"
        echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'sudo journalctl -u portfolio-backend -n 50'"
        return 1
    fi
}

# Function to display final information
show_final_info() {
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo ""
    echo "📊 Your Portfolio Backend is now running at:"
    echo "   🌐 API URL: http://$PUBLIC_IP/"
    echo "   📖 Documentation: http://$PUBLIC_IP/docs"
    echo ""
    echo "🔧 To connect via SSH:"
    echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Update your Vercel frontend to use: http://$PUBLIC_IP/"
    echo "   2. Test all API endpoints from your frontend"
    echo "   3. Consider setting up a domain name and SSL certificate"
    echo "   4. Monitor the application with: sudo journalctl -u portfolio-backend -f"
    echo ""
    echo "💰 Estimated monthly cost: \$35-50 (t3.medium instance)"
    echo ""
}

# Main execution
main() {
    create_key_pair
    create_security_group
    launch_instance
    get_public_ip
    create_deployment_package
    setup_on_ec2
    configure_services
    test_deployment
    show_final_info
    
    # Clean up deployment package
    rm portfolio-backend.tar.gz
}

# Run main function
main "$@" 