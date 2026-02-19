#!/bin/bash
# JARVIS AWS Deployment Script
# This script sets up an EC2 instance with JARVIS

set -e

echo "=========================================="
echo "JARVIS AWS Deployment"
echo "=========================================="

# Configuration
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"  # t3.medium for CPU, g4dn.xlarge for GPU
REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-jarvis-key}"
SECURITY_GROUP="${SECURITY_GROUP:-jarvis-sg}"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not installed. Install with: brew install awscli"
    exit 1
fi

# Check if configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS CLI not configured. Run: aws configure"
    exit 1
fi

echo "Using region: $REGION"
echo "Instance type: $INSTANCE_TYPE"

# Create key pair if doesn't exist
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &> /dev/null; then
    echo "Creating key pair: $KEY_NAME"
    aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" \
        --query 'KeyMaterial' --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    echo "Key saved to: ${KEY_NAME}.pem"
fi

# Create security group if doesn't exist
SG_ID=$(aws ec2 describe-security-groups --group-names "$SECURITY_GROUP" --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "")

if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
    echo "Creating security group: $SECURITY_GROUP"
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP" \
        --description "JARVIS server security group" \
        --region "$REGION" \
        --query 'GroupId' --output text)

    # Allow SSH
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --region "$REGION"

    # Allow HTTPS
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 443 --cidr 0.0.0.0/0 \
        --region "$REGION"

    # Allow HTTP (for initial setup)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 8000 --cidr 0.0.0.0/0 \
        --region "$REGION"
fi

echo "Security group ID: $SG_ID"

# Get latest Ubuntu 22.04 AMI
AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text \
    --region "$REGION")

echo "Using AMI: $AMI_ID"

# User data script to install JARVIS
USER_DATA=$(cat << 'EOF'
#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
apt-get install -y docker.io docker-compose git ffmpeg
systemctl enable docker
systemctl start docker

# Clone JARVIS
cd /home/ubuntu
git clone https://github.com/zubair2001890/jarvis.git
cd jarvis

# Create .env file (will be configured later)
cat > backend/.env << 'ENVEOF'
ANTHROPIC_API_KEY=
APP_PASSWORD=
WHISPER_MODEL=base
ENVEOF

# Build and run
docker-compose build
echo "JARVIS installed. Configure /home/ubuntu/jarvis/backend/.env and run: cd /home/ubuntu/jarvis && docker-compose up -d"

EOF
)

# Launch instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=jarvis-server}]" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=========================================="
echo "JARVIS EC2 Instance Created!"
echo "=========================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo ""
echo "Next steps:"
echo "1. Wait 2-3 minutes for setup to complete"
echo "2. SSH into the server:"
echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo ""
echo "3. Configure your API keys:"
echo "   nano /home/ubuntu/jarvis/backend/.env"
echo ""
echo "4. Start JARVIS:"
echo "   cd /home/ubuntu/jarvis && docker-compose up -d"
echo ""
echo "5. Access JARVIS:"
echo "   http://$PUBLIC_IP:8000"
echo ""
echo "For HTTPS, set up a domain and use Caddy/nginx as reverse proxy."
