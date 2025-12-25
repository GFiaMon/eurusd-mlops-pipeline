# Flask API Deployment Debug Summary
**Date**: 2025-12-25  
**Issue**: Docker container failing to start with DataManager and S3_AVAILABLE errors

## Problems Identified

### 1. Missing `utils/` Directory in Docker Image
**Error**: `CRITICAL: DataManager not found. Ensure utils/data_manager.py exists.`

**Root Cause**: The `Dockerfile.cloud` was not copying the `utils/` directory into the container, so `utils/data_manager.py` was unavailable at runtime.

**Fix**: Added `COPY utils/ /app/utils/` to Dockerfile.cloud (line 25-26)

### 2. Missing `S3_AVAILABLE` Variable
**Error**: `NameError: name 'S3_AVAILABLE' is not defined`

**Root Cause**: In `api/app_cloud.py`, the function `download_s3_json()` referenced `S3_AVAILABLE` on line 53, but this variable was never defined in the Flask app. It exists in `data_manager.py` but wasn't imported.

**Fix**: Added boto3 import with graceful fallback and S3_AVAILABLE flag definition in app_cloud.py (lines 18-26)

### 3. Missing `boto3` Import
**Root Cause**: `download_s3_json()` used `boto3.client()` without importing boto3.

**Fix**: Added boto3 import with try/except for graceful degradation (lines 19-26)

### 4. Missing `project_root` Variable
**Root Cause**: Lines 78 and 158 in `app_cloud.py` referenced `project_root` which was never defined.

**Fix**: Added project_root definition (line 28)

## Changes Made

### File: `Dockerfile.cloud`
```diff
# Copy application code
COPY api/ /app/api/

+# Copy utils directory for DataManager
+COPY utils/ /app/utils/
```

### File: `api/app_cloud.py`
```diff
# Load environment variables
load_dotenv()

+# Optional S3 support
+try:
+    import boto3
+    from botocore.exceptions import ClientError, NoCredentialsError
+    S3_AVAILABLE = True
+except ImportError:
+    S3_AVAILABLE = False
+    print("Warning: boto3 not available - S3 functionality disabled")
+
+# Project root detection
+project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
```

## Next Steps

1. **Upload the fixed files to EC2**:
   ```bash
   # From your local machine, upload the updated files
   scp -i ~/.ssh/eurusd-ml-key.pem Dockerfile.cloud ubuntu@<EC2_IP>:~/eurusd-capstone/
   scp -i ~/.ssh/eurusd-ml-key.pem api/app_cloud.py ubuntu@<EC2_IP>:~/eurusd-capstone/api/
   ```

2. **Ensure utils/ directory is on EC2**:
   ```bash
   # Make sure the entire utils/ directory is uploaded
   scp -i ~/.ssh/eurusd-ml-key.pem -r utils/ ubuntu@<EC2_IP>:~/eurusd-capstone/
   ```

3. **Re-run the deployment script on EC2**:
   ```bash
   ssh -i ~/.ssh/eurusd-ml-key.pem ubuntu@<EC2_IP>
   cd ~/eurusd-capstone
   ./scripts/deployment/deploy_flask_api.sh s3 eurusd-ml-models http://<MLFLOW_EC2_IP>:5000
   ```

4. **Verify the deployment**:
   ```bash
   # Check container logs
   docker logs -f eurusd-app
   
   # Test health endpoint
   curl http://localhost:8080/health
   ```

## Prevention

To prevent similar issues in the future:

1. **Test Docker builds locally** before deploying to EC2
2. **Use linting tools** to catch undefined variables
3. **Add integration tests** that verify all imports work correctly
4. **Document all required directories** in the Dockerfile
5. **Use explicit imports** rather than relying on module-level variables from other files

## Additional Notes

- The deployment script already has logic to copy `data_manager.py` to `api/utils/` (lines 65-80), but this was a workaround. The proper fix is to include `utils/` in the Docker image.
- boto3 should be in `api/requirements.txt` - verify this is the case
- Consider adding a `COPY utils/__init__.py /app/utils/` explicitly if there are any issues with the directory copy
