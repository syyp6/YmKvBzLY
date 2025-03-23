#!/bin/bash

# Financial Dashboard Deployment Script
# This script automates the deployment of the Financial Dashboard application to AWS

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
  echo -e "\n${YELLOW}==== $1 ====${NC}\n"
}

# Function to print success messages
print_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages and exit
print_error() {
  echo -e "${RED}✗ $1${NC}"
  exit 1
}

# Check for required tools
check_requirements() {
  print_section "Checking requirements"
  
  # Check for AWS CLI
  if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it: https://aws.amazon.com/cli/"
  fi
  print_success "AWS CLI is installed"
  
  # Check for Terraform
  if ! command -v terraform &> /dev/null; then
    print_error "Terraform is not installed. Please install it: https://www.terraform.io/downloads.html"
  fi
  print_success "Terraform is installed"
  
  # Check for Node.js and npm
  if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install it: https://nodejs.org/"
  fi
  print_success "Node.js is installed"
  
  if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install it with Node.js: https://nodejs.org/"
  fi
  print_success "npm is installed"
  
  # Check for AWS credentials
  if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials are not configured. Please run 'aws configure'"
  fi
  print_success "AWS credentials are configured"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
  print_section "Deploying infrastructure with Terraform"
  
  cd terraform/environments/dev
  
  # Initialize Terraform
  echo "Initializing Terraform..."
  terraform init || print_error "Failed to initialize Terraform"
  print_success "Terraform initialized"
  
  # Apply Terraform configuration
  echo "Applying Terraform configuration..."
  terraform apply -auto-approve || print_error "Failed to apply Terraform configuration"
  print_success "Infrastructure deployed successfully"
  
  # Extract outputs
  echo "Extracting Terraform outputs..."
  API_GATEWAY_URL=$(terraform output -raw api_gateway_url)
  COGNITO_USER_POOL_ID=$(terraform output -raw cognito_user_pool_id)
  COGNITO_CLIENT_ID=$(terraform output -raw cognito_client_id)
  CLOUDFRONT_DOMAIN=$(terraform output -raw cloudfront_distribution_domain)
  
  # Get S3 bucket name
  S3_BUCKET_NAME=$(terraform output -raw frontend_bucket_name 2>/dev/null || echo "")
  
  # If S3 bucket name is not available as a direct output, extract it from the state
  if [ -z "$S3_BUCKET_NAME" ]; then
    S3_BUCKET_NAME=$(terraform state show module.frontend.module.s3_bucket.aws_s3_bucket.this[0] | grep id | head -n 1 | awk '{print $3}' | tr -d '"')
  fi
  
  if [ -z "$S3_BUCKET_NAME" ]; then
    print_error "Failed to get S3 bucket name from Terraform outputs"
  fi
  
  print_success "Terraform outputs extracted successfully"
  
  # Return to the root directory
  cd ../../../
  
  # Export variables for later use
  export API_GATEWAY_URL
  export COGNITO_USER_POOL_ID
  export COGNITO_CLIENT_ID
  export CLOUDFRONT_DOMAIN
  export S3_BUCKET_NAME
}

# Update frontend environment variables
update_frontend_env() {
  print_section "Updating frontend environment variables"
  
  # Get AWS region from current configuration
  AWS_REGION=$(aws configure get region)
  if [ -z "$AWS_REGION" ]; then
    AWS_REGION="us-east-1"  # Default to us-east-1 if not set
  fi
  
  # Create or update .env file
  cat > src/frontend/.env << EOF
# AWS Region
REACT_APP_AWS_REGION=${AWS_REGION}

# Cognito
REACT_APP_USER_POOL_ID=${COGNITO_USER_POOL_ID}
REACT_APP_USER_POOL_CLIENT_ID=${COGNITO_CLIENT_ID}

# API Gateway
REACT_APP_API_URL=${API_GATEWAY_URL}

# Other Configuration
REACT_APP_ENV=production
EOF
  
  print_success "Frontend environment variables updated"
}

# Build frontend
build_frontend() {
  print_section "Building frontend"
  
  cd src/frontend
  
  # Install dependencies
  echo "Installing dependencies..."
  npm ci || print_error "Failed to install dependencies"
  print_success "Dependencies installed"
  
  # Build the application
  echo "Building the application..."
  npm run build || print_error "Failed to build the application"
  print_success "Frontend built successfully"
  
  # Return to the root directory
  cd ../../
}

# Deploy frontend to S3
deploy_frontend() {
  print_section "Deploying frontend to S3"
  
  echo "Uploading build files to S3..."
  aws s3 sync src/frontend/build/ s3://${S3_BUCKET_NAME}/ --delete || print_error "Failed to upload build files to S3"
  print_success "Frontend deployed to S3 successfully"
}

# Print deployment information
print_deployment_info() {
  print_section "Deployment Information"
  
  echo "Frontend URL: https://${CLOUDFRONT_DOMAIN}"
  echo "API Gateway URL: ${API_GATEWAY_URL}"
  echo "Cognito User Pool ID: ${COGNITO_USER_POOL_ID}"
  echo "Cognito Client ID: ${COGNITO_CLIENT_ID}"
  echo "S3 Bucket: ${S3_BUCKET_NAME}"
}

# Main deployment process
main() {
  print_section "Starting deployment of Financial Dashboard"
  
  check_requirements
  deploy_infrastructure
  update_frontend_env
  build_frontend
  deploy_frontend
  print_deployment_info
  
  print_section "Deployment completed successfully!"
  echo "Your Financial Dashboard is now available at: https://${CLOUDFRONT_DOMAIN}"
}

# Run the main function
main 