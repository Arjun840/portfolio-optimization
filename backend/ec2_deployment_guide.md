# EC2 Deployment Guide for Portfolio Optimization Backend

## Overview
This guide will help you deploy your FastAPI backend to Amazon EC2, which provides:
- Cost-effective hosting (cheaper than ECS)
- Unlimited storage capacity
- Full control over the environment
- Easy scaling options

## Prerequisites
- AWS CLI installed and configured
- Your Vercel frontend URL
- Domain name (optional but recommended)

## Step 1: Launch EC2 Instance

### Create Key Pair (if you don't have one)
```bash
aws ec2 create-key-pair --key-name portfolio-backend-key --query 'KeyMaterial' --output text > portfolio-backend-key.pem
chmod 400 portfolio-backend-key.pem
```

### Launch EC2 Instance
```bash
# Create security group
aws ec2 create-security-group \
    --group-name portfolio-backend-sg \
    --description "Security group for portfolio optimization backend"

# Get the security group ID
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
    --group-names portfolio-backend-sg \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

# Configure security group rules
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Launch instance (t3.medium recommended for portfolio optimization workloads)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type t3.medium \
    --key-name portfolio-backend-key \
    --security-group-ids $SECURITY_GROUP_ID \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=portfolio-backend}]'
```

### Get Instance Public IP
```bash
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=portfolio-backend" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
```

## Step 2: Connect and Setup Environment

### Connect to EC2 Instance
```bash
ssh -i portfolio-backend-key.pem ubuntu@$PUBLIC_IP
```

### Install Required Software (run on EC2 instance)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and required packages
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
sudo apt install -y nginx git curl

# Install Node.js (for potential future needs)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Create application directory
sudo mkdir -p /opt/portfolio-backend
sudo chown ubuntu:ubuntu /opt/portfolio-backend
```

## Step 3: Deploy Application Code

### Upload your code to EC2
```bash
# On your local machine, create a deployment package
cd /path/to/your/portfolio-optimization
tar -czf portfolio-backend.tar.gz backend/

# Upload to EC2
scp -i portfolio-backend-key.pem portfolio-backend.tar.gz ubuntu@$PUBLIC_IP:/opt/portfolio-backend/

# On EC2 instance, extract and setup
cd /opt/portfolio-backend
tar -xzf portfolio-backend.tar.gz
mv backend/* .
rm -rf backend portfolio-backend.tar.gz
```

### Setup Python Environment (on EC2)
```bash
cd /opt/portfolio-backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install fastapi uvicorn python-multipart
pip install pandas numpy scipy scikit-learn
pip install yfinance requests boto3
pip install python-jose[cryptography] passlib[bcrypt]
pip install matplotlib seaborn plotly

# Install additional requirements if you have a requirements.txt
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
```

## Step 4: Configure Environment Variables

### Create environment file
```bash
# On EC2 instance
cat > /opt/portfolio-backend/.env << EOF
ENVIRONMENT=production
FRONTEND_URL=https://your-vercel-app.vercel.app
SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
AWS_REGION=us-east-1
EOF

# Make sure to replace the FRONTEND_URL with your actual Vercel URL
```

## Step 5: Configure Nginx Reverse Proxy

### Create Nginx configuration
```bash
# On EC2 instance
sudo tee /etc/nginx/sites-available/portfolio-backend << EOF
server {
    listen 80;
    server_name $PUBLIC_IP;  # Replace with your domain if you have one

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
EOF

# Enable the site
sudo ln -s /etc/nginx/sites-available/portfolio-backend /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test and restart nginx
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

## Step 6: Create Systemd Service

### Create service file
```bash
# On EC2 instance
sudo tee /etc/systemd/system/portfolio-backend.service << EOF
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
EOF

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl start portfolio-backend
sudo systemctl enable portfolio-backend

# Check service status
sudo systemctl status portfolio-backend
```

## Step 7: Update CORS Configuration

### Update your API server CORS settings
Your current CORS configuration should work, but make sure to add your EC2 public IP to the frontend's API configuration.

## Step 8: Test Deployment

### Test the API
```bash
# On EC2 instance or from your local machine
curl http://$PUBLIC_IP/
curl http://$PUBLIC_IP/docs

# Test with your Vercel frontend
# Update your frontend's API base URL to: http://YOUR_EC2_PUBLIC_IP
```

## Step 9: Optional - Setup SSL with Let's Encrypt

If you have a domain name, you can set up SSL:

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add this line:
# 0 12 * * * /usr/bin/certbot renew --quiet
```

## Step 10: Update Frontend Configuration

In your Vercel frontend, update the API base URL to point to your EC2 instance:

```javascript
// In your frontend service configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'http://YOUR_EC2_PUBLIC_IP'  // or https://your-domain.com if you set up SSL
  : 'http://localhost:8000';
```

## Monitoring and Maintenance

### View logs
```bash
# Application logs
sudo journalctl -u portfolio-backend -f

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Update application
```bash
# To update your application code
cd /opt/portfolio-backend
# Upload new code and restart service
sudo systemctl restart portfolio-backend
```

## Cost Estimation
- t3.medium instance: ~$30-40/month
- Storage: ~$5-10/month
- Data transfer: Variable based on usage
- Total estimated cost: ~$35-50/month

This is significantly cheaper than ECS while providing the flexibility and storage capacity you need.

## Security Best Practices
1. Regularly update the system: `sudo apt update && sudo apt upgrade`
2. Use fail2ban for additional security: `sudo apt install fail2ban`
3. Set up CloudWatch monitoring
4. Use IAM roles instead of access keys when possible
5. Regularly backup your data

## Next Steps
1. Test all API endpoints
2. Monitor performance and adjust instance size if needed
3. Set up automated backups
4. Consider using an Application Load Balancer for high availability
5. Set up CloudWatch alarms for monitoring 