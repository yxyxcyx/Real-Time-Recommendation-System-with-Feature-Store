#!/bin/bash

# Real-Time Recommendation System Quick Start Script
# This script sets up and runs the recommendation system

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!!]${NC} $1"
}

print_error() {
    echo -e "${RED}[XX]${NC} $1"
}

# ASCII Art Banner
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Real-Time Recommendation System with Feature Store      ║
║                     Production Ready                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "Starting Quick Setup..."
echo "======================"

# Step 1: Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_status "Python $python_version detected (OK)"
else
    print_error "Python $required_version or higher required (found $python_version)"
    exit 1
fi

# Step 2: Create and activate virtual environment
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_warning "Virtual environment already exists"
fi

print_status "Activating virtual environment..."
source venv/bin/activate

# Step 3: Install dependencies
print_status "Installing dependencies (this may take a few minutes)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Step 4: Set up environment
print_status "Setting up environment..."
python main.py setup

# Step 5: Check for required services
print_warning "Checking for required services..."

# Check Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        print_status "Redis is running"
    else
        print_warning "Redis is not running. Starting Redis in Docker..."
        docker run -d -p 6379:6379 --name recsys-redis redis:latest
    fi
else
    print_warning "Redis not found. Starting Redis in Docker..."
    docker run -d -p 6379:6379 --name recsys-redis redis:latest
fi

# Step 6: Train models (optional)
read -p "Do you want to train models now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Training models..."
    python main.py train
else
    print_warning "Skipping model training. Using default models."
fi

# Step 7: Choose deployment method
echo ""
echo "How would you like to run the system?"
echo "1) Local development (separate terminals)"
echo "2) Docker Compose (all services)"
echo "3) Just the API server"

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        print_status "Starting local development setup..."
        echo ""
        echo "Please run these commands in separate terminals:"
        echo ""
        echo "Terminal 1 - API Server:"
        echo "  source venv/bin/activate && python main.py serve"
        echo ""
        echo "Terminal 2 - Stream Processor (optional):"
        echo "  source venv/bin/activate && python main.py stream"
        echo ""
        echo "Terminal 3 - Monitoring (optional):"
        echo "  docker-compose up prometheus grafana"
        ;;
    
    2)
        print_status "Starting Docker Compose deployment..."
        docker-compose up -d
        echo ""
        print_status "All services starting..."
        echo ""
        echo "Services will be available at:"
        echo "  - API:        http://localhost:8000"
        echo "  - Docs:       http://localhost:8000/docs"
        echo "  - Grafana:    http://localhost:3000 (admin/admin)"
        echo "  - MLflow:     http://localhost:5000"
        echo "  - Kafka UI:   http://localhost:8080"
        echo ""
        echo "Check status with: docker-compose ps"
        ;;
    
    3)
        print_status "Starting API server..."
        python main.py serve &
        
        sleep 5
        
        # Check if API is running
        if curl -s http://localhost:8000/health > /dev/null; then
            print_status "API server is running!"
            echo ""
            echo "API available at:"
            echo "  - Health:     http://localhost:8000/health"
            echo "  - Docs:       http://localhost:8000/docs"
            echo "  - Metrics:    http://localhost:8000/metrics"
        else
            print_error "Failed to start API server"
        fi
        ;;
    
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================"
echo "Quick Start Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Test the API:"
echo "   curl -X POST http://localhost:8000/recommend \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"user_id\": \"user_123\", \"num_recommendations\": 10}'"
echo ""
echo "2. Check system health:"
echo "   python main.py health"
echo ""
echo "3. View API documentation:"
echo "   http://localhost:8000/docs"
echo ""
print_status "Setup completed successfully!"
