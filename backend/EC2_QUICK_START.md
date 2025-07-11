# EC2 Quick Start Guide

## 🚀 Quick Deployment

### Prerequisites
1. AWS CLI installed and configured (`aws configure`)
2. Your Vercel frontend URL ready

### 1. Deploy to EC2
```bash
cd backend
./deploy_to_ec2.sh
```

This script will:
- ✅ Create EC2 instance and security groups
- ✅ Install all dependencies
- ✅ Configure nginx reverse proxy
- ✅ Set up systemd service
- ✅ Test the deployment

**Estimated time: 10-15 minutes**

### 2. Update Your Frontend
Update your Vercel frontend's API base URL to:
```
http://YOUR_EC2_PUBLIC_IP/
```

### 3. Update Application (when you make changes)
```bash
cd backend
./update_ec2.sh
```

This script will:
- ✅ Backup current deployment
- ✅ Upload new code
- ✅ Update dependencies
- ✅ Restart services
- ✅ Test the update

**Estimated time: 2-3 minutes**

## 📊 Cost Breakdown
- **t3.medium instance**: ~$35-40/month
- **Storage**: ~$5-10/month
- **Data transfer**: Variable
- **Total**: ~$40-50/month

**Much cheaper than ECS while maintaining full control!**

## 🔧 Common Commands

### Check Status
```bash
ssh -i portfolio-backend-key.pem ubuntu@YOUR_EC2_IP
sudo systemctl status portfolio-backend
```

### View Logs
```bash
ssh -i portfolio-backend-key.pem ubuntu@YOUR_EC2_IP
sudo journalctl -u portfolio-backend -f
```

### Restart Service
```bash
ssh -i portfolio-backend-key.pem ubuntu@YOUR_EC2_IP
sudo systemctl restart portfolio-backend
```

## 🛡️ Security Notes
- SSH access is restricted to your IP
- Nginx handles CORS properly
- Environment variables are secured
- Regular backups are created automatically

## 🚨 Troubleshooting

### API Not Responding
```bash
# Check service status
ssh -i portfolio-backend-key.pem ubuntu@YOUR_EC2_IP 'sudo systemctl status portfolio-backend'

# Check logs
ssh -i portfolio-backend-key.pem ubuntu@YOUR_EC2_IP 'sudo journalctl -u portfolio-backend -n 50'

# Restart if needed
ssh -i portfolio-backend-key.pem ubuntu@YOUR_EC2_IP 'sudo systemctl restart portfolio-backend'
```

### CORS Issues
If you have CORS issues, update the environment variable:
```bash
ssh -i portfolio-backend-key.pem ubuntu@YOUR_EC2_IP
cd /opt/portfolio-backend
sudo nano .env
# Update FRONTEND_URL to your correct Vercel URL
sudo systemctl restart portfolio-backend
```

### High CPU Usage
Consider upgrading to a larger instance:
```bash
# Stop instance
aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID

# Modify instance type
aws ec2 modify-instance-attribute --instance-id YOUR_INSTANCE_ID --instance-type t3.large

# Start instance
aws ec2 start-instances --instance-ids YOUR_INSTANCE_ID
```

## 📈 Scaling Options
- **Vertical**: Upgrade instance type (t3.medium → t3.large → t3.xlarge)
- **Horizontal**: Add Application Load Balancer + Auto Scaling Group
- **Storage**: Add EBS volumes for data persistence
- **Performance**: Use CloudFront CDN for static assets

## 🔄 CI/CD Integration
You can integrate the update script with GitHub Actions:

```yaml
name: Deploy to EC2
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to EC2
        run: |
          echo "${{ secrets.EC2_PRIVATE_KEY }}" > portfolio-backend-key.pem
          chmod 400 portfolio-backend-key.pem
          cd backend
          ./update_ec2.sh
```

## 📞 Support
If you encounter issues:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Ensure your AWS credentials are properly configured
4. Verify your security group settings allow the necessary ports 