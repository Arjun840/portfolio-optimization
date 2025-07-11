#!/bin/bash

# Portfolio Backend BUDGET EC2 Deployment Script
# This script deploys your FastAPI backend to a budget-friendly EC2 instance (~$5-10/month)

set -e

echo "💰 Starting BUDGET Portfolio Backend EC2 Deployment..."
echo "💡 This deployment targets ~$5-10/month cost"

# Configuration
KEY_NAME="portfolio-backend-key"
SECURITY_GROUP_NAME="portfolio-backend-sg"
INSTANCE_NAME="portfolio-backend"

# Budget instance options
echo ""
echo "💰 Choose your budget instance type:"
echo "1) t2.micro  - $8-9/month  (FREE for 12 months if new AWS user!)"
echo "2) t3.micro  - $7-8/month  (Better performance)"
echo "3) t3.nano   - $4-5/month  (Minimal resources)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        INSTANCE_TYPE="t2.micro"
        echo "✅ Selected t2.micro (~$8/month, FREE if eligible for free tier)"
        ;;
    2)
        INSTANCE_TYPE="t3.micro"
        echo "✅ Selected t3.micro (~$7/month)"
        ;;
    3)
        INSTANCE_TYPE="t3.nano"
        echo "✅ Selected t3.nano (~$4/month - minimal resources)"
        echo "⚠️  Warning: t3.nano may be slow for portfolio optimization"
        ;;
    *)
        echo "❌ Invalid choice. Defaulting to t2.micro (eligible for free tier)"
        INSTANCE_TYPE="t2.micro"
        ;;
esac

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS CLI is configured"

# Check if user might be eligible for free tier
echo ""
echo "💡 TIP: If you're a new AWS user, t2.micro is FREE for 12 months!"
echo "   Check your AWS Free Tier eligibility at: https://aws.amazon.com/free/"

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
        aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
        aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
        aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 443 --cidr 0.0.0.0/0
        aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0
        
        echo "✅ Security group configured: $SECURITY_GROUP_ID"
    fi
}

# Function to launch EC2 instance
launch_instance() {
    EXISTING_INSTANCE=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$EXISTING_INSTANCE" != "None" ] && [ "$EXISTING_INSTANCE" != "null" ]; then
        echo "✅ Instance already exists: $EXISTING_INSTANCE"
        INSTANCE_ID=$EXISTING_INSTANCE
    else
        echo "🚀 Launching budget EC2 instance ($INSTANCE_TYPE)..."
        
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

# Function to create lightweight deployment package
create_deployment_package() {
    echo "📦 Creating lightweight deployment package..."
    tar -czf portfolio-backend-budget.tar.gz \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.log' \
        --exclude='analysis_plots' \
        --exclude='lambda-*' \
        --exclude='*.zip' \
        --exclude='data/individual_assets' \
        --exclude='*.png' \
        --exclude='*.jpg' \
        .
    echo "✅ Lightweight deployment package created"
}

# Function to setup on EC2 with performance optimizations for small instances
setup_on_ec2() {
    echo "📤 Uploading code to budget EC2 instance..."
    
    # Wait for SSH to be available
    echo "⏳ Waiting for SSH to be available..."
    while ! ssh -i ${KEY_NAME}.pem -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP echo "SSH Ready" > /dev/null 2>&1; do
        sleep 10
        echo "   Still waiting for SSH..."
    done
    echo "✅ SSH is ready"
    
    # Upload deployment package
    scp -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no portfolio-backend-budget.tar.gz ubuntu@$PUBLIC_IP:/tmp/
    
    # Run setup commands on EC2 with memory optimization
    ssh -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'EOF'
        set -e
        echo "🔧 Setting up budget server environment..."
        
        # Update system
        sudo apt update && sudo apt upgrade -y
        
        # Install required packages
        sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip nginx git curl htop
        
        # Create swap file for small instances (improves performance)
        if [ ! -f /swapfile ]; then
            echo "💾 Creating swap file for better performance..."
            sudo fallocate -l 1G /swapfile
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
            echo "✅ Swap file created"
        fi
        
        # Create application directory
        sudo mkdir -p /opt/portfolio-backend
        sudo chown ubuntu:ubuntu /opt/portfolio-backend
        
        # Extract application code
        cd /opt/portfolio-backend
        tar -xzf /tmp/portfolio-backend-budget.tar.gz
        rm /tmp/portfolio-backend-budget.tar.gz
        
        # Create virtual environment
        python3.11 -m venv venv
        source venv/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip
        
        # Install core dependencies only (to save memory and startup time)
        echo "📦 Installing lightweight dependencies..."
        pip install fastapi==0.104.1 uvicorn[standard]==0.23.2
        pip install pandas numpy scipy scikit-learn
        pip install yfinance requests
        pip install python-jose[cryptography] passlib[bcrypt]
        
        # Install additional requirements if available
        if [ -f requirements_ec2.txt ]; then
            pip install -r requirements_ec2.txt
        fi
        
        echo "✅ Lightweight Python environment set up successfully"
EOF
    
    echo "✅ Code uploaded and environment configured for budget instance"
}

# Function to configure services with performance optimizations
configure_services() {
    echo "⚙️  Configuring services for budget instance..."
    
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
        
        # Configure Nginx with optimizations for small instances
        sudo tee /etc/nginx/sites-available/portfolio-backend << 'NGINXCONF'
server {
    listen 80;
    server_name _;
    
    # Optimize for small instances
    client_max_body_size 10M;
    keepalive_timeout 30;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 30;
        
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
        
        # Create systemd service optimized for budget instances (single worker)
        sudo tee /etc/systemd/system/portfolio-backend.service << 'SERVICECONF'
[Unit]
Description=Portfolio Optimization Backend (Budget)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/portfolio-backend
Environment=PATH=/opt/portfolio-backend/venv/bin
EnvironmentFile=/opt/portfolio-backend/.env
ExecStart=/opt/portfolio-backend/venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICECONF
        
        # Start services
        sudo systemctl daemon-reload
        sudo systemctl start portfolio-backend
        sudo systemctl enable portfolio-backend
        
        echo "✅ Services configured for budget instance"
EOF
    
    echo "✅ Services configured successfully"
}

# Function to test deployment
test_deployment() {
    echo "🧪 Testing budget deployment..."
    
    if curl -f -s --connect-timeout 15 "http://$PUBLIC_IP/" > /dev/null; then
        echo "✅ Budget API is responding at http://$PUBLIC_IP/"
        echo "📖 API documentation available at: http://$PUBLIC_IP/docs"
    else
        echo "❌ API is not responding. This might be normal for small instances (they can be slow to start)"
        echo "🔧 Wait a few minutes and check manually: http://$PUBLIC_IP/"
    fi
}

# Function to display final information
show_final_info() {
    echo ""
    echo "🎉 Budget deployment completed successfully!"
    echo ""
    echo "💰 Your Portfolio Backend is now running at BUDGET COST:"
    echo "   🌐 API URL: http://$PUBLIC_IP/"
    echo "   📖 Documentation: http://$PUBLIC_IP/docs"
    echo ""
    case $INSTANCE_TYPE in
        "t2.micro")
            echo "💰 Monthly Cost: FREE for 12 months (if new AWS user), then ~$8/month"
            ;;
        "t3.micro")
            echo "💰 Monthly Cost: ~$7-8/month"
            ;;
        "t3.nano")
            echo "💰 Monthly Cost: ~$4-5/month"
            ;;
    esac
    echo ""
    echo "⚠️  Performance Notes for Budget Instance:"
    echo "   • API responses may be slower than larger instances"
    echo "   • Portfolio optimization may take longer"
    echo "   • Perfect for personal use and demos"
    echo "   • Can upgrade to larger instance anytime if needed"
    echo ""
    echo "🔧 To upgrade later:"
    echo "   aws ec2 stop-instances --instance-ids $INSTANCE_ID"
    echo "   aws ec2 modify-instance-attribute --instance-id $INSTANCE_ID --instance-type t3.small"
    echo "   aws ec2 start-instances --instance-ids $INSTANCE_ID"
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
    rm portfolio-backend-budget.tar.gz
}

# Run main function
main "$@" 