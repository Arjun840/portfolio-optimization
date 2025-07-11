# 🚀 Portfolio Backend Deployment Options

## 💰 Cost Comparison Overview

| Option | Monthly Cost | Setup Command | Best For |
|--------|-------------|---------------|----------|
| **AWS Free Tier (t2.micro)** | **FREE** → $8 | `./deploy_to_ec2_budget.sh` | **✅ Recommended** |
| AWS Lightsail | **$5** | `./deploy_to_lightsail.sh` | Fixed pricing |
| Budget EC2 (t3.nano) | $4-5 | `./deploy_to_ec2_budget.sh` | Ultra-cheap |
| Budget EC2 (t3.micro) | $7-8 | `./deploy_to_ec2_budget.sh` | Good performance |
| Standard EC2 (t3.medium) | $35-40 | `./deploy_to_ec2.sh` | Production ready |

## 🎯 Recommendations by Use Case

### 🆓 **New AWS Users → FREE TIER**
```bash
cd backend
./deploy_to_ec2_budget.sh
# Choose option 1: t2.micro (FREE for 12 months!)
```
- **Cost**: FREE for 12 months
- **After**: $8/month
- **Perfect for**: Personal projects, learning, portfolio demos

### 💰 **Budget-Conscious → LIGHTSAIL**
```bash
cd backend
./deploy_to_lightsail.sh
```
- **Cost**: Fixed $5/month (includes bandwidth)
- **Perfect for**: Predictable billing, simple management
- **Includes**: 1 vCPU, 1GB RAM, 40GB storage, 1TB transfer

### ⚡ **Performance on Budget → t3.micro**
```bash
cd backend
./deploy_to_ec2_budget.sh
# Choose option 2: t3.micro
```
- **Cost**: $7-8/month
- **Perfect for**: Better performance than nano, still budget-friendly

### 🚀 **Production Ready → t3.medium**
```bash
cd backend
./deploy_to_ec2.sh
```
- **Cost**: $35-40/month
- **Perfect for**: Multiple users, fast optimization, production use

## 📊 Detailed Comparison

### 🆓 **AWS Free Tier (t2.micro)**
**✅ BEST CHOICE for new AWS users**

- **Cost**: FREE for 12 months, then $8/month
- **Specs**: 1 vCPU, 1GB RAM, 30GB storage
- **Performance**: Good for personal use
- **Limitations**: Only for new AWS accounts
- **Includes**: 750 hours/month (24/7), 30GB storage, 15GB bandwidth

### ⚡ **AWS Lightsail**
**✅ BEST for predictable costs**

- **Cost**: Fixed $5/month
- **Specs**: 1 vCPU, 1GB RAM, 40GB SSD storage
- **Performance**: Similar to t2.micro
- **Advantages**: Fixed billing, includes bandwidth, simple management
- **Includes**: 1TB data transfer included

### 💸 **EC2 Budget Options**
**✅ MOST flexible**

| Instance | vCPU | RAM | Monthly Cost | Performance |
|----------|------|-----|-------------|-------------|
| t3.nano | 2 | 0.5GB | $4-5 | Slow |
| t3.micro | 2 | 1GB | $7-8 | Good |
| t3.small | 2 | 2GB | $16-20 | Fast |
| t3.medium | 2 | 4GB | $35-40 | Very Fast |

## 🔧 Quick Setup Commands

### Deploy FREE (if eligible):
```bash
cd backend
./deploy_to_ec2_budget.sh
# Choose: t2.micro (FREE for 12 months)
```

### Deploy $5/month (Lightsail):
```bash
cd backend  
./deploy_to_lightsail.sh
```

### Deploy $4-8/month (Budget EC2):
```bash
cd backend
./deploy_to_ec2_budget.sh
# Choose your preferred instance size
```

## 📈 Performance Expectations

### Budget Instances (FREE - $8/month):
- **API Response Time**: 2-5 seconds
- **Portfolio Optimization**: 10-30 seconds
- **Concurrent Users**: 1-3
- **Perfect for**: Personal use, demos, development

### Standard Instance ($35-40/month):
- **API Response Time**: 1-2 seconds
- **Portfolio Optimization**: 5-15 seconds
- **Concurrent Users**: 10+
- **Perfect for**: Production, multiple users

## 🚨 Important Notes

### Free Tier Eligibility:
- ✅ New AWS accounts only
- ✅ 750 hours/month for 12 months
- ✅ 30GB EBS storage included
- ⚠️ After 12 months: $8/month

### All Options Include:
- ✅ Automatic deployment
- ✅ CORS configured for Vercel
- ✅ SSL-ready
- ✅ Auto-restart on crashes
- ✅ Easy updates
- ✅ Monitoring capabilities

## 🔄 Easy Upgrades

All deployments can be upgraded later:

```bash
# Stop instance
aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID

# Upgrade to larger size
aws ec2 modify-instance-attribute --instance-id YOUR_INSTANCE_ID --instance-type t3.small

# Start instance
aws ec2 start-instances --instance-ids YOUR_INSTANCE_ID
```

## 🛠️ Management Commands

### Check Status:
```bash
cd backend
./check_ec2_status.sh
```

### Update Code:
```bash
cd backend
./update_ec2.sh
```

### Clean Up Lambda Files:
```bash
./cleanup_lambda_files.sh
```

## 💡 Final Recommendation

**For most users**: Start with the **AWS Free Tier** (t2.micro) using:
```bash
cd backend
./deploy_to_ec2_budget.sh
```

- It's **FREE for 12 months**
- Perfect for personal portfolio projects
- Easy to upgrade later if needed
- Handles portfolio optimization well for personal use

**Only choose paid options if**:
- You're not eligible for free tier
- You want predictable billing (Lightsail)
- You need better performance immediately

All scripts are automated and will have your backend running in 10-15 minutes! 🎉 