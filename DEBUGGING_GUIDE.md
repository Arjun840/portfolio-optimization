# 🐛 Debugging Guide - Dependency Conflicts

## 🚨 **The Problem You Encountered**

You hit a **dependency conflict** during deployment:

```
ERROR: Cannot install fastapi==0.104.1 and typing-extensions==4.7.1 because these package versions have conflicting dependencies.

The conflict is caused by:
    The user requested typing-extensions==4.7.1
    fastapi 0.104.1 depends on typing-extensions>=4.8.0
```

### **What This Means:**
- FastAPI 0.104.1 requires `typing-extensions>=4.8.0` (version 4.8.0 or higher)
- The requirements file specified `typing-extensions==4.7.1` (exactly version 4.7.1)
- These versions are incompatible, so pip couldn't install both

## 🔧 **How to Fix This**

### **Option 1: Fix Existing Instance (Quick Fix)**

If you already have an EC2 instance running with this error:

```bash
# Upload the fix script to your EC2 instance
scp -i portfolio-backend-key.pem backend/fix_dependencies.sh ubuntu@YOUR_EC2_IP:/tmp/

# SSH into your instance
ssh -i portfolio-backend-key.pem ubuntu@YOUR_EC2_IP

# Run the fix script
chmod +x /tmp/fix_dependencies.sh
/tmp/fix_dependencies.sh
```

### **Option 2: Fresh Deployment (Recommended)**

Use the corrected deployment script:

```bash
cd backend
./deploy_to_ec2_budget_fixed.sh
```

This script now uses `requirements_ec2_fixed.txt` which has compatible versions.

## 📋 **What the Fix Does**

### **Before (Broken):**
```
fastapi==0.104.1           # Needs typing-extensions>=4.8.0
typing-extensions==4.7.1   # Too old! Conflict!
```

### **After (Fixed):**
```
fastapi==0.104.1           # Needs typing-extensions>=4.8.0  
typing-extensions>=4.8.0   # Compatible! ✅
```

## 🔍 **How to Debug Similar Issues**

### **Check Package Dependencies:**
```bash
# See what version is installed
pip show typing-extensions

# See what FastAPI requires
pip show fastapi

# Check for conflicts
pip check
```

### **Fix Dependency Conflicts:**
```bash
# Method 1: Let pip resolve automatically
pip install fastapi uvicorn --upgrade

# Method 2: Install specific compatible versions
pip install "typing-extensions>=4.8.0" "fastapi==0.104.1"

# Method 3: Use requirements file without version pins
pip install -r requirements_flexible.txt
```

## 🛠️ **Common Python Dependency Issues on EC2**

### **Issue 1: Python Version Mismatch**
```
E: Unable to locate package python3.11
```
**Fix:** Use default Python version (3.10 on Ubuntu 22.04)

### **Issue 2: Package Conflicts**
```
ERROR: Cannot install X and Y because of conflicting dependencies
```
**Fix:** Check version compatibility, use flexible version ranges

### **Issue 3: Outdated pip**
```
WARNING: You are using pip version X.X.X; however, version Y.Y.Y is available
```
**Fix:** Always upgrade pip first: `pip install --upgrade pip`

### **Issue 4: Missing System Dependencies**
```
error: Microsoft Visual C++ 14.0 is required
```
**Fix:** Install system packages: `sudo apt install python3-dev build-essential`

## 📦 **Requirements File Best Practices**

### **❌ Too Restrictive (Causes Conflicts):**
```
fastapi==0.104.1
typing-extensions==4.7.1
pydantic==2.4.0
```

### **✅ Flexible but Safe:**
```
fastapi==0.104.1
typing-extensions>=4.8.0
pydantic>=2.4.0
```

### **✅ Most Flexible:**
```
fastapi>=0.100.0,<1.0.0
typing-extensions>=4.8.0
pydantic>=2.0.0,<3.0.0
```

## 🔄 **Prevention Strategies**

### **1. Test Requirements Locally:**
```bash
# Create clean virtual environment
python -m venv test_venv
source test_venv/bin/activate

# Test your requirements file
pip install -r requirements.txt

# Check for conflicts
pip check
```

### **2. Use Dependency Management Tools:**
```bash
# Use pipenv for better dependency resolution
pipenv install fastapi uvicorn

# Or use poetry
poetry add fastapi uvicorn
```

### **3. Pin Major Versions Only:**
Instead of `fastapi==0.104.1`, use `fastapi>=0.104.0,<1.0.0`

## 📊 **Debugging Commands Reference**

### **Check Service Status:**
```bash
sudo systemctl status portfolio-backend
sudo systemctl is-active portfolio-backend
```

### **View Logs:**
```bash
# Recent logs
sudo journalctl -u portfolio-backend -n 50

# Live logs
sudo journalctl -u portfolio-backend -f

# All logs since boot
sudo journalctl -u portfolio-backend --since today
```

### **Test API:**
```bash
# Test locally on EC2
curl http://localhost:8000/

# Test from your machine
curl http://YOUR_EC2_IP/

# Test with timeout
curl --connect-timeout 10 http://YOUR_EC2_IP/
```

### **Python Environment:**
```bash
# Check Python version
python3 --version

# List installed packages
pip list

# Check specific package
pip show fastapi

# Check for conflicts
pip check
```

## ✅ **Verification Steps**

After fixing, verify everything works:

1. **Service is running:**
   ```bash
   sudo systemctl status portfolio-backend
   ```

2. **API responds:**
   ```bash
   curl http://localhost:8000/
   ```

3. **No dependency conflicts:**
   ```bash
   cd /opt/portfolio-backend
   source venv/bin/activate
   pip check
   ```

4. **Logs are clean:**
   ```bash
   sudo journalctl -u portfolio-backend -n 20
   ```

## 🆘 **If Problems Persist**

### **Nuclear Option - Complete Reset:**
```bash
# Stop service
sudo systemctl stop portfolio-backend

# Remove virtual environment
sudo rm -rf /opt/portfolio-backend/venv

# Recreate from scratch
cd /opt/portfolio-backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_ec2_fixed.txt

# Restart service
sudo systemctl start portfolio-backend
```

### **Check System Resources:**
```bash
# Memory usage
free -h

# Disk space
df -h

# CPU usage
top

# System load
uptime
```

## 💡 **Pro Tips**

1. **Always upgrade pip first:** `pip install --upgrade pip`
2. **Use virtual environments:** Never install packages globally
3. **Test locally first:** Verify requirements work on your machine
4. **Keep backups:** Save working requirements files
5. **Monitor logs:** Use `journalctl -f` to watch for issues
6. **Start simple:** Install core packages first, add features incrementally

The dependency conflict you encountered is common in Python deployment. The fixes I've provided should resolve the issue and prevent it in the future! 🎉 