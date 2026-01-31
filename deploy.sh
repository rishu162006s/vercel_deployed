#!/bin/bash
echo "🚀 Deploying AI Battle PDF QA System to Vercel..."

# Create project structure
mkdir -p api
cp aibattle.py api/
cp requirements.txt .
cp vercel.json .
cp .env .

# Deploy to Vercel
echo "📦 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo "🌍 Your API endpoint: https://your-project.vercel.app/aibattle"
echo "📚 API Docs: https://your-project.vercel.app/docs"
