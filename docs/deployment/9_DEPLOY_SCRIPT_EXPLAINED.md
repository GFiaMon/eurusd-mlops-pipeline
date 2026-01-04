# 📜 The `deploy.sh` Script Explained

## 🤔 Why use it?
You asked: *"Can i use just that instead all the different manual steps to setup the ecs with flask on docker?"*

**The short answer is YES, but with one caveat.**

The script automates **Step 2** of the process. You still need to do **Step 1** manually (or with `scp`).

### The Two Steps of Deployment

1.  **Transfer Files (Step 1)**: You must get your code onto the EC2 server. The script *cannot* do this for you because it runs *inside* the server (usually).
2.  **Run Application (Step 2)**: Once code is there, you need to build the Docker image, stop any old containers, and start the new one with the right ports and environment variables. **This is what `deploy.sh` does.**

---

## 🛠️ What specific manual steps does it replace?

Without the script, every time you updated your app, you would have to type all this manually into the terminal:

```bash
# 1. Stop the old running container (if it exists)
docker stop eurusd-app

# 2. Remove the old container
docker rm eurusd-app

# 3. Re-build the new image
docker build -t eurusd-predictor:latest .

# 4. Run the new container (with a crazy long command!)
docker run -d \
    --name eurusd-app \
    -p 80:8080 \
    -e USE_S3=false \
    -e PORT=8080 \
    --restart unless-stopped \
    eurusd-predictor:latest
```

**With the script, you just type:**
```bash
./scripts/deploy.sh
```

It replaces ~4 complex commands with 1 simple command. It prevents typos and "forgetting" arguments (like exposing the port).

---

## 📂 How to use it with the `scripts/` folder

Since we moved it to `scripts/`, here is the updated workflow:

### 1. On your local machine (Transfer)
You still need to copy the files to the server.
```bash
# Create package (including the new scripts folder)
tar -czf eurusd-app.tar.gz \
    api/ \
    scripts/ \
    models/lstm_trained_model.keras \
    models/lstm_scaler.joblib \
    data/processed/lstm_simple_test_data.csv \
    Dockerfile .dockerignore

# Upload
scp eurusd-app.tar.gz eurusd:~/
```

### 2. On the EC2 Server (Run)
```bash
# Unzip
tar -xzf eurusd-app.tar.gz

# Make executable (only needed once)
chmod +x scripts/deploy.sh

# Run
./scripts/deploy.sh
```

---

## 📝 Script Breakdown (Line by Line)

Here is exactly what the code does:

```bash
# Finds where the script is, so it knows where your Dockerfile is
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Stops the old version of your app so ports don't clash
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Builds the new version from your source code
cd "$PROJECT_ROOT"
docker build -t $DOCKER_IMAGE .

# Runs the new container
# It checks if you ran "./scripts/deploy.sh s3" or just "./scripts/deploy.sh"
if [ "$STORAGE_TYPE" = "s3" ]; then
   # Runs with S3 variables
   docker run ...
else
   # Runs with Local storage variables
   docker run ...
fi
```
