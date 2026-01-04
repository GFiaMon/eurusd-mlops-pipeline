# Quick Start Guide - Docker Deployment

## 🚀 Quick Deploy (Local Testing)

### Option 1: Using Docker Compose (Easiest)
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Using Deploy Script
```bash
# For local storage (EBS)
./deploy.sh local

# For S3 storage
./deploy.sh s3 your-bucket-name
```

### Option 3: Manual Docker Commands
```bash
# Build
docker build -t eurusd-predictor .

# Run with local storage
docker run -d --name eurusd-app -p 8080:8080 eurusd-predictor

# Run with S3 storage
docker run -d --name eurusd-app -p 8080:8080 \
    -e USE_S3=true \
    -e S3_BUCKET=your-bucket-name \
    eurusd-predictor
```

---

## 📦 What's Included

### Files Created
1. **Dockerfile** - Container definition
2. **docker-compose.yml** - Easy local testing
3. **deploy.sh** - Automated deployment script
4. **api/app_cloud.py** - Enhanced app with S3 support
5. **api/requirements.txt** - Updated dependencies
6. **.dockerignore** - Optimized builds
7. **docs/AWS_DEPLOYMENT_GUIDE.md** - Complete deployment guide

---

## 🔧 Testing Locally

### 1. Test Docker Build
```bash
docker build -t eurusd-predictor .
```

### 2. Test Local Storage
```bash
docker run -p 8080:8080 eurusd-predictor
# Visit: http://localhost:8080
```

### 3. Test S3 Storage (if configured)
```bash
docker run -p 8080:8080 \
    -e USE_S3=true \
    -e S3_BUCKET=your-bucket \
    -v ~/.aws:/root/.aws:ro \
    eurusd-predictor
```

---

## ☁️ AWS EC2 Deployment

### Quick Deploy to EC2

1. **Transfer files to EC2:**
```bash
# Create deployment package
tar -czf eurusd-app.tar.gz api/ models/ data/ Dockerfile .dockerignore deploy.sh

# Upload to EC2
scp -i your-key.pem eurusd-app.tar.gz ec2-user@your-ec2-ip:~/
```

2. **On EC2, extract and deploy:**
```bash
tar -xzf eurusd-app.tar.gz
./deploy.sh local  # or: ./deploy.sh s3 your-bucket-name
```

---

## 🎯 Which Storage Option?

### Use **Local Storage (EBS)** if:
- ✅ Single EC2 instance
- ✅ Development/testing
- ✅ Want simplicity
- ✅ Models < 10GB
- **No code changes needed - use current app.py**

### Use **S3 Storage** if:
- ✅ Multiple EC2 instances
- ✅ Production deployment
- ✅ Need to update models frequently
- ✅ Want automatic backups
- **Use app_cloud.py instead of app.py**

---

## 📝 Code Changes Summary

### For Local Storage (EBS)
**No changes needed!** Your current `api/app.py` works perfectly.

Just update Dockerfile if needed:
```dockerfile
CMD ["python", "api/app.py"]
```

### For S3 Storage
Update Dockerfile to use the cloud version:
```dockerfile
CMD ["python", "api/app_cloud.py"]
```

Set environment variables when running:
```bash
-e USE_S3=true
-e S3_BUCKET=your-bucket-name
```

---

## 🔍 Monitoring

### Check if app is running
```bash
docker ps
curl http://localhost:8080/health
```

### View logs
```bash
docker logs -f eurusd-app
```

### Check resource usage
```bash
docker stats eurusd-app
```

---

## 🛠️ Troubleshooting

### Container exits immediately
```bash
docker logs eurusd-app
# Check for missing models or data files
```

### Out of memory
```bash
# Use larger EC2 instance (t3.medium minimum)
# Or add swap space (see deployment guide)
```

### Can't access from browser
```bash
# Check security group allows port 80
# Check container is running: docker ps
# Check logs: docker logs eurusd-app
```

---

## 📊 Cost Comparison

| Component | EBS | S3 |
|-----------|-----|-----|
| EC2 (t3.medium) | $30/mo | $30/mo |
| Storage | $3/mo | $0.12/mo |
| Simplicity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Scalability | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Best for** | Dev/Test | Production |

---

## 🎓 Next Steps

1. **Test locally** with Docker
2. **Choose storage approach** (EBS or S3)
3. **Deploy to EC2**
4. **Set up monitoring**
5. **Add CI/CD** (optional)

📖 For detailed instructions, see: `docs/AWS_DEPLOYMENT_GUIDE.md`
