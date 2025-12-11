# ✅ Deployment Checklist

## 📋 Pre-Deployment Checklist

### Local Testing
- [ ] Docker Desktop installed and running
- [ ] Test build: `docker build -t eurusd-predictor .`
- [ ] Test run: `docker-compose up`
- [ ] Verify app at `http://localhost:8080`
- [ ] Test `/health` endpoint
- [ ] Test `/api/predict` endpoint
- [ ] Test `/api/stats` endpoint
- [ ] Verify predictions are working

### AWS Prerequisites
- [ ] AWS account created
- [ ] AWS CLI installed and configured
- [ ] EC2 key pair created and downloaded
- [ ] Decided on storage option (EBS or S3)

---

## 🚀 EBS Deployment Checklist

### Step 1: Prepare Files
- [ ] Navigate to project directory
- [ ] Create deployment package:
  ```bash
  tar -czf eurusd-app.tar.gz \
      api/ \
      models/lstm_trained_model.keras \
      models/lstm_scaler.joblib \
      data/processed/lstm_simple_test_data.csv \
      Dockerfile .dockerignore deploy.sh
  ```
- [ ] Verify package size (should be ~100-500MB)

### Step 2: Launch EC2 Instance
- [ ] Instance type: **t3.medium** or larger
- [ ] AMI: **Amazon Linux 2023** or Ubuntu 22.04
- [ ] Storage: **20-30 GB** EBS gp3
- [ ] Security Group created with rules:
  - [ ] Port 80 (HTTP) - Source: 0.0.0.0/0
  - [ ] Port 22 (SSH) - Source: Your IP
- [ ] Key pair attached
- [ ] Instance launched and running
- [ ] Note public IP address: ________________

### Step 3: Install Docker on EC2
- [ ] SSH to EC2: `ssh -i key.pem ec2-user@<IP>`
- [ ] Update system: `sudo yum update -y`
- [ ] Install Docker: `sudo yum install -y docker`
- [ ] Start Docker: `sudo systemctl start docker`
- [ ] Enable Docker: `sudo systemctl enable docker`
- [ ] Add user to docker group: `sudo usermod -a -G docker ec2-user`
- [ ] Log out and back in
- [ ] Verify Docker: `docker --version`

### Step 4: Transfer Files
- [ ] From local machine, upload package:
  ```bash
  scp -i key.pem eurusd-app.tar.gz ec2-user@<IP>:~/
  ```
- [ ] Verify upload completed

### Step 5: Deploy Application
- [ ] SSH to EC2
- [ ] Extract files: `tar -xzf eurusd-app.tar.gz`
- [ ] Make deploy script executable: `chmod +x deploy.sh`
- [ ] Deploy: `./deploy.sh local`
- [ ] Wait for deployment to complete
- [ ] Check container is running: `docker ps`
- [ ] Check logs: `docker logs eurusd-app`

### Step 6: Verify Deployment
- [ ] Test health: `curl http://localhost/health`
- [ ] Test from browser: `http://<EC2-PUBLIC-IP>`
- [ ] Verify predictions are working
- [ ] Check all endpoints:
  - [ ] `/` (home page)
  - [ ] `/api/predict`
  - [ ] `/api/stats`
  - [ ] `/health`

### Step 7: Optional Enhancements
- [ ] Set up Elastic IP (static IP)
- [ ] Configure domain name
- [ ] Add SSL certificate (Let's Encrypt)
- [ ] Set up CloudWatch monitoring
- [ ] Configure automatic backups (EBS snapshots)

---

## ☁️ S3 Deployment Checklist

### Step 1: Create S3 Bucket
- [ ] Create bucket: `aws s3 mb s3://eurusd-ml-models`
- [ ] Bucket name: ________________
- [ ] Region: ________________

### Step 2: Upload Models to S3
- [ ] Upload model:
  ```bash
  aws s3 cp models/lstm_trained_model.keras \
      s3://eurusd-ml-models/models/lstm_trained_model.keras
  ```
- [ ] Upload scaler:
  ```bash
  aws s3 cp models/lstm_scaler.joblib \
      s3://eurusd-ml-models/models/lstm_scaler.joblib
  ```
- [ ] Upload data:
  ```bash
  aws s3 cp data/processed/lstm_simple_test_data.csv \
      s3://eurusd-ml-models/data/processed/lstm_simple_test_data.csv
  ```
- [ ] Verify uploads: `aws s3 ls s3://eurusd-ml-models/ --recursive`

### Step 3: Create IAM Role
- [ ] Create trust policy file (see deployment guide)
- [ ] Create IAM role: `eurusd-ec2-s3-role`
- [ ] Create S3 access policy (see deployment guide)
- [ ] Attach policy to role
- [ ] Create instance profile: `eurusd-ec2-profile`
- [ ] Add role to instance profile

### Step 4: Launch EC2 with IAM Role
- [ ] Instance type: **t3.medium** or larger
- [ ] AMI: **Amazon Linux 2023** or Ubuntu 22.04
- [ ] Storage: **10-20 GB** EBS gp3 (smaller than EBS option)
- [ ] **IAM instance profile:** eurusd-ec2-profile
- [ ] Security Group with rules:
  - [ ] Port 80 (HTTP) - Source: 0.0.0.0/0
  - [ ] Port 22 (SSH) - Source: Your IP
- [ ] Instance launched and running
- [ ] Note public IP: ________________

### Step 5: Prepare Deployment Package
- [ ] Update Dockerfile CMD to use `app_cloud.py`:
  ```dockerfile
  CMD ["python", "api/app_cloud.py"]
  ```
- [ ] Create package (no models/data needed):
  ```bash
  tar -czf eurusd-app.tar.gz \
      api/ Dockerfile .dockerignore deploy.sh
  ```
- [ ] Package should be much smaller (~50MB)

### Step 6: Install Docker on EC2
- [ ] Same as EBS steps 3 (see above)

### Step 7: Transfer and Deploy
- [ ] Upload package: `scp -i key.pem eurusd-app.tar.gz ec2-user@<IP>:~/`
- [ ] SSH to EC2
- [ ] Extract: `tar -xzf eurusd-app.tar.gz`
- [ ] Deploy with S3:
  ```bash
  ./deploy.sh s3 eurusd-ml-models
  ```
- [ ] Wait for deployment (will download from S3)
- [ ] Check logs: `docker logs eurusd-app`

### Step 8: Verify S3 Deployment
- [ ] Check S3 downloads in logs
- [ ] Test health: `curl http://localhost/health`
- [ ] Verify response shows `"storage_type": "S3"`
- [ ] Test from browser: `http://<EC2-PUBLIC-IP>`
- [ ] Verify predictions working

### Step 9: Test Model Update (S3 Only)
- [ ] Upload new model to S3
- [ ] Restart container: `docker restart eurusd-app`
- [ ] Verify new model downloaded
- [ ] Test predictions still work

---

## 🔍 Troubleshooting Checklist

### Container Won't Start
- [ ] Check logs: `docker logs eurusd-app`
- [ ] Check if models exist (EBS): `ls -lh models/`
- [ ] Check S3 access (S3): `aws s3 ls s3://eurusd-ml-models/`
- [ ] Check IAM role attached (S3): `aws sts get-caller-identity`
- [ ] Try running interactively: `docker run -it eurusd-predictor bash`

### Can't Access from Browser
- [ ] Check container running: `docker ps`
- [ ] Check from EC2: `curl http://localhost`
- [ ] Check Security Group allows port 80
- [ ] Check using public IP (not private)
- [ ] Check firewall on EC2

### Out of Memory
- [ ] Check instance type (minimum t3.medium)
- [ ] Check memory usage: `docker stats`
- [ ] Add swap space (see deployment guide)
- [ ] Consider larger instance type

### S3 Access Denied
- [ ] Verify IAM role attached to EC2
- [ ] Test S3 access: `aws s3 ls s3://eurusd-ml-models/`
- [ ] Check IAM policy permissions
- [ ] Verify bucket name is correct

### Predictions Not Working
- [ ] Check model files loaded: `docker logs eurusd-app | grep "Model loaded"`
- [ ] Check data loaded: `docker logs eurusd-app | grep "Data loaded"`
- [ ] Test API directly: `curl http://localhost/api/predict`
- [ ] Check for errors in logs

---

## 📊 Post-Deployment Checklist

### Monitoring
- [ ] Set up CloudWatch logs
- [ ] Configure CloudWatch alarms
- [ ] Set up health check monitoring
- [ ] Configure email alerts

### Security
- [ ] Review Security Group rules
- [ ] Disable SSH from 0.0.0.0/0 (use your IP only)
- [ ] Set up VPC if needed
- [ ] Enable S3 bucket encryption (S3 option)
- [ ] Enable S3 versioning (S3 option)

### Backup
- [ ] Configure EBS snapshots (EBS option)
- [ ] Enable S3 versioning (S3 option)
- [ ] Document backup procedure
- [ ] Test restore procedure

### Documentation
- [ ] Document EC2 instance details
- [ ] Document S3 bucket name (if using S3)
- [ ] Document IAM role name (if using S3)
- [ ] Document any custom configurations
- [ ] Update project README

### Performance
- [ ] Load test the application
- [ ] Monitor response times
- [ ] Check memory usage
- [ ] Consider auto-scaling if needed

---

## 🎯 Quick Command Reference

### Docker Commands
```bash
# View logs
docker logs -f eurusd-app

# Restart app
docker restart eurusd-app

# Stop app
docker stop eurusd-app

# Remove container
docker rm eurusd-app

# Check stats
docker stats eurusd-app

# Shell into container
docker exec -it eurusd-app bash
```

### EC2 Commands
```bash
# SSH to EC2
ssh -i key.pem ec2-user@<IP>

# Check disk space
df -h

# Check memory
free -h

# Check processes
top

# View system logs
sudo journalctl -u docker
```

### S3 Commands
```bash
# List bucket
aws s3 ls s3://eurusd-ml-models/ --recursive

# Upload file
aws s3 cp local-file s3://bucket/path/

# Download file
aws s3 cp s3://bucket/path/ local-file

# Sync directory
aws s3 sync models/ s3://bucket/models/
```

---

## ✅ Success Criteria

Your deployment is successful when:
- [ ] Container is running: `docker ps` shows eurusd-app
- [ ] Health check passes: `/health` returns 200
- [ ] Home page loads in browser
- [ ] Predictions are working: `/api/predict` returns valid predictions
- [ ] Stats endpoint works: `/api/stats` returns data
- [ ] No errors in logs
- [ ] Response time < 1 second
- [ ] App survives container restart

---

## 📞 Need Help?

**Documentation:**
- `README_DEPLOYMENT.md` - Start here
- `DEPLOYMENT_SUMMARY.md` - File explanations
- `STORAGE_COMPARISON.md` - EBS vs S3
- `docs/AWS_DEPLOYMENT_GUIDE.md` - Detailed guide
- `ARCHITECTURE.md` - Architecture diagrams

**Common Issues:**
See Troubleshooting section above

---

**Good luck with your deployment! 🚀**

Print this checklist and check off items as you complete them!
