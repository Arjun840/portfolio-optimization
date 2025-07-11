#!/bin/bash

# Portfolio Backend EC2 Status Check Script
# This script checks the status of your deployed FastAPI backend on EC2

set -e

echo "📊 Checking Portfolio Backend EC2 Status..."

# Configuration
KEY_NAME="portfolio-backend-key"
INSTANCE_NAME="portfolio-backend"

# Function to get instance information
get_instance_info() {
    INSTANCE_ID=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,pending,stopping,stopped" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$INSTANCE_ID" = "None" ] || [ "$INSTANCE_ID" = "null" ]; then
        echo "❌ No instance found with name: $INSTANCE_NAME"
        echo "Please run the deployment script first: ./deploy_to_ec2.sh"
        exit 1
    fi
    
    INSTANCE_STATE=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text)
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    INSTANCE_TYPE=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].InstanceType' \
        --output text)
    
    LAUNCH_TIME=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].LaunchTime' \
        --output text)
    
    echo "✅ Instance ID: $INSTANCE_ID"
    echo "🔄 State: $INSTANCE_STATE"
    echo "🌐 Public IP: $PUBLIC_IP"
    echo "💻 Instance Type: $INSTANCE_TYPE"
    echo "🕒 Launch Time: $LAUNCH_TIME"
}

# Function to check API health
check_api_health() {
    if [ "$INSTANCE_STATE" != "running" ]; then
        echo "⚠️  Instance is not running, cannot check API health"
        return
    fi
    
    echo ""
    echo "🔍 Checking API Health..."
    
    # Test basic connectivity
    if curl -f -s --connect-timeout 10 "http://$PUBLIC_IP/" > /dev/null 2>&1; then
        echo "✅ API is responding"
        
        # Get API info
        API_INFO=$(curl -s "http://$PUBLIC_IP/" 2>/dev/null || echo "{}")
        if [ "$API_INFO" != "{}" ]; then
            echo "📋 API Information:"
            echo "$API_INFO" | python3 -m json.tool 2>/dev/null || echo "$API_INFO"
        fi
        
        echo ""
        echo "🔗 Available URLs:"
        echo "   📖 API Documentation: http://$PUBLIC_IP/docs"
        echo "   🔧 Alternative Docs: http://$PUBLIC_IP/redoc"
        echo "   ❤️  Health Check: http://$PUBLIC_IP/"
        
    else
        echo "❌ API is not responding"
        echo "🔧 Troubleshooting commands:"
        echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'sudo systemctl status portfolio-backend'"
        echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'sudo journalctl -u portfolio-backend -n 20'"
    fi
}

# Function to check service status on EC2
check_service_status() {
    if [ "$INSTANCE_STATE" != "running" ]; then
        echo "⚠️  Instance is not running, cannot check service status"
        return
    fi
    
    echo ""
    echo "🔧 Checking Service Status on EC2..."
    
    if [ ! -f "${KEY_NAME}.pem" ]; then
        echo "❌ Key file ${KEY_NAME}.pem not found"
        return
    fi
    
    ssh -i ${KEY_NAME}.pem -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'EOF' 2>/dev/null || echo "❌ Could not connect to EC2 instance"
        echo "📊 System Status:"
        echo "   Uptime: $(uptime)"
        echo "   Disk Usage: $(df -h /opt/portfolio-backend | tail -1 | awk '{print $5}') used"
        echo "   Memory Usage: $(free -h | grep '^Mem:' | awk '{print $3"/"$2}')"
        echo ""
        
        echo "🚀 Portfolio Backend Service:"
        if sudo systemctl is-active --quiet portfolio-backend; then
            echo "   Status: ✅ Active (Running)"
        else
            echo "   Status: ❌ Inactive/Failed"
        fi
        
        echo "   Since: $(sudo systemctl show portfolio-backend --property=ActiveEnterTimestamp --value)"
        echo ""
        
        echo "🌐 Nginx Service:"
        if sudo systemctl is-active --quiet nginx; then
            echo "   Status: ✅ Active (Running)"
        else
            echo "   Status: ❌ Inactive/Failed"
        fi
        echo ""
        
        echo "📁 Application Directory:"
        echo "   Size: $(du -sh /opt/portfolio-backend 2>/dev/null | cut -f1)"
        echo "   Files: $(find /opt/portfolio-backend -type f | wc -l) files"
        echo ""
        
        echo "📝 Recent Logs (last 5 lines):"
        sudo journalctl -u portfolio-backend -n 5 --no-pager
EOF
}

# Function to show cost estimation
show_cost_info() {
    echo ""
    echo "💰 Cost Estimation:"
    
    if [ "$INSTANCE_TYPE" = "t3.medium" ]; then
        echo "   Instance (t3.medium): ~$35-40/month"
    elif [ "$INSTANCE_TYPE" = "t3.large" ]; then
        echo "   Instance (t3.large): ~$60-70/month"
    elif [ "$INSTANCE_TYPE" = "t3.small" ]; then
        echo "   Instance (t3.small): ~$15-20/month"
    else
        echo "   Instance ($INSTANCE_TYPE): Check AWS pricing"
    fi
    
    echo "   Storage (20GB EBS): ~$2-3/month"
    echo "   Data Transfer: Variable based on usage"
    echo "   Total Estimated: Check your AWS billing dashboard"
}

# Function to show management commands
show_management_commands() {
    if [ "$INSTANCE_STATE" != "running" ]; then
        echo ""
        echo "🔧 Instance Management:"
        echo "   Start: aws ec2 start-instances --instance-ids $INSTANCE_ID"
        echo "   Stop: aws ec2 stop-instances --instance-ids $INSTANCE_ID"
        echo "   Terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
        return
    fi
    
    echo ""
    echo "🔧 Useful Commands:"
    echo "   Connect via SSH:"
    echo "     ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
    echo ""
    echo "   Update deployment:"
    echo "     ./update_ec2.sh"
    echo ""
    echo "   View live logs:"
    echo "     ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'sudo journalctl -u portfolio-backend -f'"
    echo ""
    echo "   Restart service:"
    echo "     ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'sudo systemctl restart portfolio-backend'"
}

# Main execution
main() {
    get_instance_info
    check_api_health
    check_service_status
    show_cost_info
    show_management_commands
    
    echo ""
    echo "✨ Status check completed!"
}

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install AWS CLI first."
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Run main function
main "$@" 