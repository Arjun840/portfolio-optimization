#!/bin/bash

# Portfolio Backend EC2 Update Script
# This script updates your deployed FastAPI backend on EC2

set -e

echo "🔄 Starting Portfolio Backend EC2 Update..."

# Configuration
KEY_NAME="portfolio-backend-key"
INSTANCE_NAME="portfolio-backend"

# Function to get instance information
get_instance_info() {
    INSTANCE_ID=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$INSTANCE_ID" = "None" ] || [ "$INSTANCE_ID" = "null" ]; then
        echo "❌ No running instance found with name: $INSTANCE_NAME"
        echo "Please run the deployment script first: ./deploy_to_ec2.sh"
        exit 1
    fi
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo "✅ Found instance: $INSTANCE_ID"
    echo "🌐 Public IP: $PUBLIC_IP"
}

# Function to create deployment package
create_deployment_package() {
    echo "📦 Creating deployment package..."
    tar -czf portfolio-backend-update.tar.gz \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.log' \
        --exclude='analysis_plots' \
        --exclude='lambda-*' \
        --exclude='*.zip' \
        --exclude='portfolio-backend-update.tar.gz' \
        .
    echo "✅ Deployment package created: portfolio-backend-update.tar.gz"
}

# Function to backup current deployment
backup_current_deployment() {
    echo "💾 Creating backup of current deployment..."
    ssh -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'EOF'
        cd /opt/portfolio-backend
        timestamp=$(date +%Y%m%d_%H%M%S)
        sudo tar -czf "/opt/portfolio-backend-backup-$timestamp.tar.gz" . 2>/dev/null || true
        echo "✅ Backup created: /opt/portfolio-backend-backup-$timestamp.tar.gz"
EOF
}

# Function to update application
update_application() {
    echo "📤 Uploading updated code..."
    
    # Upload new deployment package
    scp -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no portfolio-backend-update.tar.gz ubuntu@$PUBLIC_IP:/tmp/
    
    # Update application on EC2
    ssh -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'EOF'
        set -e
        echo "🔄 Updating application..."
        
        # Stop the service
        sudo systemctl stop portfolio-backend
        
        # Extract new code
        cd /opt/portfolio-backend
        tar -xzf /tmp/portfolio-backend-update.tar.gz
        rm /tmp/portfolio-backend-update.tar.gz
        
        # Activate virtual environment and update dependencies
        source venv/bin/activate
        pip install --upgrade pip
        
        # Install/update requirements if they exist
        if [ -f requirements_ec2.txt ]; then
            pip install -r requirements_ec2.txt
        elif [ -f requirements.txt ]; then
            pip install -r requirements.txt
        fi
        
        # Restart the service
        sudo systemctl start portfolio-backend
        
        # Check service status
        if sudo systemctl is-active --quiet portfolio-backend; then
            echo "✅ Application updated and restarted successfully"
        else
            echo "❌ Application failed to start. Check logs:"
            sudo journalctl -u portfolio-backend -n 20
            exit 1
        fi
EOF
}

# Function to test updated deployment
test_deployment() {
    echo "🧪 Testing updated deployment..."
    
    # Wait a moment for the service to fully start
    sleep 5
    
    # Test API health
    if curl -f -s "http://$PUBLIC_IP/" > /dev/null; then
        echo "✅ Updated API is responding at http://$PUBLIC_IP/"
        echo "📖 API documentation available at: http://$PUBLIC_IP/docs"
    else
        echo "❌ Updated API is not responding. Check the logs:"
        echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'sudo journalctl -u portfolio-backend -n 50'"
        return 1
    fi
}

# Function to display update information
show_update_info() {
    echo ""
    echo "🎉 Update completed successfully!"
    echo ""
    echo "📊 Your Portfolio Backend is running the latest code at:"
    echo "   🌐 API URL: http://$PUBLIC_IP/"
    echo "   📖 Documentation: http://$PUBLIC_IP/docs"
    echo ""
    echo "🔧 To check logs:"
    echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'sudo journalctl -u portfolio-backend -f'"
    echo ""
    echo "🗂️  To rollback if needed:"
    echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
    echo "   cd /opt"
    echo "   ls portfolio-backend-backup-*.tar.gz  # Find your backup"
    echo "   # Then extract the backup over the current deployment"
    echo ""
}

# Main execution
main() {
    get_instance_info
    create_deployment_package
    backup_current_deployment
    update_application
    test_deployment
    show_update_info
    
    # Clean up deployment package
    rm portfolio-backend-update.tar.gz
}

# Check if key file exists
if [ ! -f "${KEY_NAME}.pem" ]; then
    echo "❌ Key file ${KEY_NAME}.pem not found."
    echo "Please ensure you're running this script from the same directory where you ran the deployment script."
    exit 1
fi

# Run main function
main "$@" 