#!/bin/bash

# Portfolio Backend AWS Lightsail Deployment Script
# AWS Lightsail offers fixed pricing: $5/month for 1GB RAM, 1 vCPU, 40GB SSD

set -e

echo "⚡ Starting Portfolio Backend AWS Lightsail Deployment..."
echo "💰 Fixed cost: $5/month (includes bandwidth allowance)"

# Configuration
INSTANCE_NAME="portfolio-backend-lightsail"
BUNDLE_ID="nano_2_0"  # $5/month: 1 vCPU, 1GB RAM, 40GB SSD
BLUEPRINT_ID="ubuntu_22_04"
AVAILABILITY_ZONE="us-east-1a"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS CLI is configured"

# Function to create key pair if it doesn't exist
create_key_pair() {
    if ! aws lightsail get-key-pair --key-pair-name $INSTANCE_NAME > /dev/null 2>&1; then
        echo "🔑 Creating Lightsail key pair..."
        aws lightsail create-key-pair --key-pair-name $INSTANCE_NAME --query 'keyPair.privateKey' --output text > ${INSTANCE_NAME}.pem
        chmod 400 ${INSTANCE_NAME}.pem
        echo "✅ Key pair created: ${INSTANCE_NAME}.pem"
    else
        echo "✅ Key pair already exists"
        # Download private key if we don't have it locally
        if [ ! -f "${INSTANCE_NAME}.pem" ]; then
            echo "📥 Downloading existing private key..."
            aws lightsail download-default-key-pair --query 'privateKeyBase64' --output text | base64 -d > ${INSTANCE_NAME}.pem
            chmod 400 ${INSTANCE_NAME}.pem
        fi
    fi
}

# Function to create Lightsail instance
create_instance() {
    # Check if instance already exists
    if aws lightsail get-instance --instance-name $INSTANCE_NAME > /dev/null 2>&1; then
        echo "✅ Lightsail instance already exists: $INSTANCE_NAME"
        INSTANCE_STATE=$(aws lightsail get-instance --instance-name $INSTANCE_NAME --query 'instance.state.name' --output text)
        echo "🔄 Instance state: $INSTANCE_STATE"
        
        if [ "$INSTANCE_STATE" != "running" ]; then
            echo "⏳ Waiting for instance to be running..."
            while [ "$INSTANCE_STATE" != "running" ]; do
                sleep 10
                INSTANCE_STATE=$(aws lightsail get-instance --instance-name $INSTANCE_NAME --query 'instance.state.name' --output text)
                echo "   Current state: $INSTANCE_STATE"
            done
        fi
    else
        echo "🚀 Creating Lightsail instance ($5/month)..."
        
        # Create user data script for initial setup
        cat > user_data.sh << 'EOF'
#!/bin/bash
apt update && apt upgrade -y
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip nginx git curl htop
systemctl enable nginx
systemctl start nginx
EOF
        
        aws lightsail create-instances \
            --instance-names $INSTANCE_NAME \
            --availability-zone $AVAILABILITY_ZONE \
            --blueprint-id $BLUEPRINT_ID \
            --bundle-id $BUNDLE_ID \
            --key-pair-name $INSTANCE_NAME \
            --user-data file://user_data.sh
        
        rm user_data.sh
        
        echo "⏳ Waiting for instance to be running..."
        while true; do
            INSTANCE_STATE=$(aws lightsail get-instance --instance-name $INSTANCE_NAME --query 'instance.state.name' --output text 2>/dev/null || echo "pending")
            echo "   Current state: $INSTANCE_STATE"
            if [ "$INSTANCE_STATE" = "running" ]; then
                break
            fi
            sleep 15
        done
        echo "✅ Instance is running"
    fi
}

# Function to configure firewall
configure_firewall() {
    echo "🛡️  Configuring firewall..."
    
    # Open HTTP port
    aws lightsail put-instance-public-ports \
        --instance-name $INSTANCE_NAME \
        --port-infos fromPort=80,toPort=80,protocol=TCP,accessFrom=0.0.0.0/0 \
                    fromPort=443,toPort=443,protocol=TCP,accessFrom=0.0.0.0/0 \
                    fromPort=22,toPort=22,protocol=TCP,accessFrom=0.0.0.0/0 \
                    fromPort=8000,toPort=8000,protocol=TCP,accessFrom=0.0.0.0/0
    
    echo "✅ Firewall configured"
}

# Function to get public IP
get_public_ip() {
    PUBLIC_IP=$(aws lightsail get-instance --instance-name $INSTANCE_NAME --query 'instance.publicIpAddress' --output text)
    echo "🌐 Public IP: $PUBLIC_IP"
}

# Function to create deployment package
create_deployment_package() {
    echo "📦 Creating deployment package for Lightsail..."
    tar -czf portfolio-backend-lightsail.tar.gz \
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
    echo "✅ Deployment package created"
}

# Function to setup application on Lightsail
setup_on_lightsail() {
    echo "📤 Setting up application on Lightsail..."
    
    # Wait for SSH to be available
    echo "⏳ Waiting for SSH to be available..."
    while ! ssh -i ${INSTANCE_NAME}.pem -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP echo "SSH Ready" > /dev/null 2>&1; do
        sleep 10
        echo "   Still waiting for SSH..."
    done
    echo "✅ SSH is ready"
    
    # Upload deployment package
    scp -i ${INSTANCE_NAME}.pem -o StrictHostKeyChecking=no portfolio-backend-lightsail.tar.gz ubuntu@$PUBLIC_IP:/tmp/
    
    # Setup application
    ssh -i ${INSTANCE_NAME}.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'EOF'
        set -e
        echo "🔧 Setting up application on Lightsail..."
        
        # Wait for user data script to complete
        while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
            echo "   Waiting for system updates to complete..."
            sleep 10
        done
        
        # Create swap file for better performance
        if [ ! -f /swapfile ]; then
            echo "💾 Creating swap file..."
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
        tar -xzf /tmp/portfolio-backend-lightsail.tar.gz
        rm /tmp/portfolio-backend-lightsail.tar.gz
        
        # Create virtual environment
        python3.11 -m venv venv
        source venv/bin/activate
        
        # Install dependencies
        pip install --upgrade pip
        pip install fastapi==0.104.1 uvicorn[standard]==0.23.2
        pip install pandas numpy scipy scikit-learn
        pip install yfinance requests
        pip install python-jose[cryptography] passlib[bcrypt]
        
        # Install additional requirements if available
        if [ -f requirements_ec2.txt ]; then
            pip install -r requirements_ec2.txt
        fi
        
        echo "✅ Application setup completed"
EOF
    
    echo "✅ Application deployed to Lightsail"
}

# Function to configure services
configure_services() {
    echo "⚙️  Configuring services..."
    
    # Prompt for Vercel URL
    echo "Please enter your Vercel frontend URL (e.g., https://your-app.vercel.app):"
    read -r FRONTEND_URL
    
    ssh -i ${INSTANCE_NAME}.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << EOF
        set -e
        cd /opt/portfolio-backend
        
        # Create environment file
        cat > .env << EOL
ENVIRONMENT=production
FRONTEND_URL=$FRONTEND_URL
SECRET_KEY=\$(openssl rand -hex 32)
JWT_SECRET_KEY=\$(openssl rand -hex 32)
AWS_REGION=us-east-1
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
        
        # Enable site
        sudo ln -sf /etc/nginx/sites-available/portfolio-backend /etc/nginx/sites-enabled/
        sudo rm -f /etc/nginx/sites-enabled/default
        sudo nginx -t
        sudo systemctl restart nginx
        
        # Create systemd service
        sudo tee /etc/systemd/system/portfolio-backend.service << 'SERVICECONF'
[Unit]
Description=Portfolio Optimization Backend (Lightsail)
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
        
        # Start service
        sudo systemctl daemon-reload
        sudo systemctl start portfolio-backend
        sudo systemctl enable portfolio-backend
        
        echo "✅ Services configured"
EOF
    
    echo "✅ Services configured successfully"
}

# Function to test deployment
test_deployment() {
    echo "🧪 Testing Lightsail deployment..."
    
    if curl -f -s --connect-timeout 15 "http://$PUBLIC_IP/" > /dev/null; then
        echo "✅ API is responding at http://$PUBLIC_IP/"
        echo "📖 API documentation: http://$PUBLIC_IP/docs"
    else
        echo "❌ API not responding yet. This is normal - wait a few minutes for startup."
        echo "🔧 Manual check: http://$PUBLIC_IP/"
    fi
}

# Function to show final information
show_final_info() {
    echo ""
    echo "🎉 Lightsail deployment completed!"
    echo ""
    echo "⚡ Your Portfolio Backend on AWS Lightsail:"
    echo "   🌐 API URL: http://$PUBLIC_IP/"
    echo "   📖 Documentation: http://$PUBLIC_IP/docs"
    echo ""
    echo "💰 Fixed Monthly Cost: $5.00 USD"
    echo "   ✅ Includes: 1 vCPU, 1GB RAM, 40GB SSD"
    echo "   ✅ Includes: 1TB data transfer"
    echo "   ✅ Predictable billing"
    echo ""
    echo "🔧 Management:"
    echo "   Connect: ssh -i ${INSTANCE_NAME}.pem ubuntu@$PUBLIC_IP"
    echo "   Console: https://lightsail.aws.amazon.com/"
    echo "   Logs: ssh -i ${INSTANCE_NAME}.pem ubuntu@$PUBLIC_IP 'sudo journalctl -u portfolio-backend -f'"
    echo ""
    echo "📊 Performance: Perfect for personal use and demos"
    echo "🎯 Upgrade: Can scale to higher bundles anytime"
    echo ""
}

# Main execution
main() {
    create_key_pair
    create_instance
    configure_firewall
    get_public_ip
    create_deployment_package
    setup_on_lightsail
    configure_services
    test_deployment
    show_final_info
    
    # Cleanup
    rm portfolio-backend-lightsail.tar.gz
}

# Check if Lightsail is available in region
echo "🔍 Checking Lightsail availability..."
if ! aws lightsail get-regions > /dev/null 2>&1; then
    echo "❌ Lightsail not available or AWS CLI needs updating"
    echo "💡 Try: aws configure set region us-east-1"
    exit 1
fi

echo "✅ Lightsail is available"

# Run main function
main "$@" 