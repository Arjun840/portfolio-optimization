# Production Setup Guide

## ✅ Current Status
- **Backend**: http://54.149.156.84/ (EC2, $5/month)
- **Frontend**: Deploying to Vercel (automatic)
- **Integration**: Updated API configuration

## 🧪 Testing Your Full Application

### 1. Wait for Vercel Deployment (2-3 minutes)
Check your Vercel dashboard for deployment status

### 2. Test Frontend Integration
Visit your Vercel app and test:
- [ ] Sign up for new account
- [ ] Log in with existing account
- [ ] View available stocks
- [ ] Run portfolio optimization
- [ ] Save/load portfolios
- [ ] View efficient frontier charts

### 3. Test API Endpoints Directly
```bash
# Health check
curl http://54.149.156.84/

# API documentation
open http://54.149.156.84/docs
```

## 🚀 Optional Production Enhancements

### A. Set Up SSL/HTTPS (Recommended)
**Cost**: ~$0.50/month for AWS Certificate Manager

1. **Purchase a domain** (optional, ~$12/year)
2. **Set up SSL certificate** via AWS Certificate Manager
3. **Configure Application Load Balancer**

### B. Set Up Custom Domain
If you have a domain:
1. **Backend**: Create A record pointing to `54.149.156.84`
2. **Frontend**: Configure custom domain in Vercel

### C. Enhanced Monitoring
```bash
# Set up CloudWatch monitoring
# Monitor EC2 instance performance
# Set up alerts for high CPU/memory usage
```

### D. Backup & Security
- [ ] Set up automated EC2 snapshots
- [ ] Configure fail2ban for SSH protection
- [ ] Set up log rotation

## 🔧 Maintenance Commands

### Backend Management
```bash
# View logs
ssh -i backend/portfolio-backend-key.pem ubuntu@54.149.156.84 'sudo journalctl -u portfolio-backend -f'

# Restart service
ssh -i backend/portfolio-backend-key.pem ubuntu@54.149.156.84 'sudo systemctl restart portfolio-backend'

# Update code (after changes)
cd backend && ./update_ec2.sh
```

### Scaling Options
If you need better performance:
1. **Upgrade instance**: `t3.micro` ($7/month) → `t3.small` ($15/month)
2. **Add load balancer**: For high availability
3. **Database**: Move from SQLite to RDS PostgreSQL

## 💰 Current Costs
- **Backend**: FREE (12 months), then $8/month
- **Frontend**: FREE (Vercel hobby plan)
- **Total**: **$0** for first year, then **$8/month**

## 🎉 You're Done!
Once Vercel deployment completes and you've tested the integration, your portfolio optimization app is fully deployed and ready for production use!

## 🆘 Troubleshooting
If you encounter issues:
1. Check Vercel deployment logs
2. Test backend API directly: http://54.149.156.84/docs
3. Check browser network tab for CORS errors
4. Verify frontend is using correct API URL

Contact if you need assistance with any of these steps. 