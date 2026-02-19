# JARVIS AWS Deployment

## Quick Start

### Option 1: Automated Script
```bash
# Configure AWS CLI first
aws configure

# Run deployment script
./deploy.sh
```

### Option 2: Manual Setup

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04
   - Instance type: `t3.medium` (CPU) or `g4dn.xlarge` (GPU)
   - Storage: 30GB minimum
   - Security group: Open ports 22, 443, 8000

2. **SSH into instance**
   ```bash
   ssh -i your-key.pem ubuntu@<public-ip>
   ```

3. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install -y docker.io docker-compose git ffmpeg
   sudo systemctl enable docker
   sudo systemctl start docker
   sudo usermod -aG docker ubuntu
   # Log out and back in
   ```

4. **Clone and configure**
   ```bash
   git clone https://github.com/zubair2001890/jarvis.git
   cd jarvis
   cp backend/.env.example backend/.env
   nano backend/.env  # Add your keys
   ```

5. **Build and run**
   ```bash
   docker-compose up -d --build
   ```

6. **Access JARVIS**
   - http://<public-ip>:8000

## Instance Type Recommendations

| Type | vCPU | RAM | GPU | Transcription Speed | Cost/hr |
|------|------|-----|-----|---------------------|---------|
| t3.medium | 2 | 4GB | No | ~5-10s per chunk | ~$0.04 |
| t3.large | 2 | 8GB | No | ~3-5s per chunk | ~$0.08 |
| g4dn.xlarge | 4 | 16GB | Yes | ~1-2s per chunk | ~$0.50 |

## HTTPS Setup (Recommended)

1. Get a domain and point it to your EC2 IP

2. Install Caddy:
   ```bash
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update
   sudo apt install caddy
   ```

3. Configure Caddy:
   ```bash
   sudo nano /etc/caddy/Caddyfile
   ```
   ```
   your-domain.com {
       reverse_proxy localhost:8000
   }
   ```

4. Restart Caddy:
   ```bash
   sudo systemctl restart caddy
   ```

## Security Notes

- Always set `APP_PASSWORD` in production
- Use HTTPS in production (Caddy handles this automatically)
- Restrict security group to your IP if possible
- Audio never leaves your server (transcribed locally with Whisper)
- Only text analysis goes to Anthropic API
