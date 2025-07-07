from mangum import Mangum
from api_server import app

# Lambda handler
handler = Mangum(app)

# Optional: Add Lambda-specific environment setup
import os
import boto3

def setup_aws_environment():
    """Configure AWS-specific settings"""
    # Set up AWS SDK clients
    if 'AWS_REGION' in os.environ:
        # Configure any AWS services you need
        pass
    
    # Set production environment variables
    os.environ.setdefault('SECRET_KEY', os.environ.get('JWT_SECRET_KEY', 'your-secret-key'))
    
# Initialize on cold start
setup_aws_environment() 