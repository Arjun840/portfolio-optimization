# 💰 Budget Deployment Guide

## 🎯 Perfect for Personal Projects & Demos

If you can't afford $40-50/month, here are budget-friendly options starting at **FREE** or as low as **$4/month**!

## 🆓 FREE Option (Best Choice!)

### AWS Free Tier - t2.micro
- **Cost**: 100% FREE for 12 months (new AWS users)
- **Specs**: 1 vCPU, 1GB RAM
- **Perfect for**: Personal portfolio projects, learning, demos
- **After 12 months**: ~$8/month

**✅ Recommended for most users**

## 💵 Paid Budget Options

### 1. t3.nano (~$4-5/month)
- **Cheapest** paid option
- 2 vCPUs, 0.5GB RAM
- Suitable for light usage
- ⚠️ May be slow for complex optimizations

### 2. t3.micro (~$7-8/month)
- Good balance of cost/performance  
- 2 vCPUs, 1GB RAM
- Handles portfolio optimization well

### 3. t2.micro (~$8-9/month)
- Similar to t3.micro
- Burstable performance
- Good for intermittent usage

## 🚀 Deploy Budget Version

### One Command Deployment:
```bash
cd backend
./deploy_to_ec2_budget.sh
```

The script will ask you to choose:
1. **t2.micro** (FREE for new users!)
2. **t3.micro** ($7/month)  
3. **t3.nano** ($4/month)

## 💡 Budget Optimizations Included

### Performance Optimizations:
- ✅ 1GB swap file for better memory management
- ✅ Single worker process (saves RAM)
- ✅ Lightweight package installation
- ✅ Optimized nginx configuration
- ✅ Excludes heavy analysis plots from deployment

### Cost Optimizations:
- ✅ Uses smallest viable instance types
- ✅ Minimizes data transfer
- ✅ Efficient resource usage
- ✅ Auto-stop capability

## 📊 Performance Expectations

### Budget Instance Performance:
- **API responses**: 2-5 seconds (vs 1-2 seconds on larger instances)
- **Portfolio optimization**: 10-30 seconds (vs 5-15 seconds)
- **Concurrent users**: 1-3 (vs 10+ on larger instances)
- **Perfect for**: Personal use, demos, development

### When to Upgrade:
- Multiple concurrent users
- Need faster optimization
- High traffic volume
- Production with SLA requirements

## 🔧 Easy Upgrade Path

You can upgrade anytime without losing data:

```bash
# Get your instance ID
aws ec2 describe-instances --filters "Name=tag:Name,Values=portfolio-backend"

# Stop instance
aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID

# Upgrade to larger instance
aws ec2 modify-instance-attribute --instance-id YOUR_INSTANCE_ID --instance-type t3.small

# Start instance
aws ec2 start-instances --instance-ids YOUR_INSTANCE_ID
```

## 💰 Cost Comparison

| Option | Monthly Cost | RAM | Performance | Best For |
|--------|-------------|-----|-------------|----------|
| **t2.micro (Free Tier)** | **FREE** → $8 | 1GB | Good | **✅ Recommended** |
| t3.nano | $4-5 | 0.5GB | Slow | Minimal usage |
| t3.micro | $7-8 | 1GB | Good | Regular usage |
| t3.small | $16-20 | 2GB | Fast | Growth stage |
| t3.medium (original) | $35-40 | 4GB | Very Fast | Production |

## 🆓 Check Free Tier Eligibility

Visit: https://aws.amazon.com/free/

**Free for 12 months includes:**
- 750 hours/month of t2.micro (enough for 24/7 usage)
- 30GB of EBS storage
- 15GB of bandwidth

## 🛠️ Budget Deployment Steps

1. **Check eligibility**: Are you a new AWS user? → Use t2.micro for FREE!
2. **Deploy**: `./deploy_to_ec2_budget.sh`
3. **Choose instance**: Select based on your needs and budget
4. **Update frontend**: Use the provided EC2 IP address
5. **Monitor costs**: Set up billing alerts in AWS console

## ⚡ Performance Tips for Budget Instances

### Frontend Optimizations:
- Add loading states for API calls
- Cache portfolio results
- Implement request debouncing
- Show progress indicators

### Backend Optimizations (Already Included):
- Single worker process
- Memory-efficient data loading
- Optimized package selection
- Swap file for memory overflow

## 🔍 Monitoring Budget Usage

### Set up billing alerts:
1. Go to AWS Billing Dashboard
2. Set alert for $5-10/month
3. Monitor Free Tier usage

### Check costs:
```bash
# View current month's costs
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-31 --granularity MONTHLY --metrics BlendedCost
```

## 🚨 Important Notes

### Free Tier Limitations:
- ⏰ Only 750 hours/month (≈ 24/7 for one instance)
- 📅 Only for first 12 months
- 🆕 Only for new AWS accounts

### Budget Instance Limitations:
- 🐌 Slower performance than larger instances
- 👤 Best for 1-3 concurrent users
- ⚠️ May timeout on very large optimizations

### Recommended Usage:
- ✅ Personal portfolios
- ✅ Demos and prototypes  
- ✅ Learning and development
- ✅ Low-traffic applications

## 🎉 Get Started

Ready to deploy for FREE or cheap? Run:

```bash
cd backend
./deploy_to_ec2_budget.sh
```

The script handles everything automatically and will have your backend running in 10-15 minutes!

Perfect for getting your portfolio optimization API online without breaking the bank! 💪 