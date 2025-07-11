#!/bin/bash

echo "🔒 Setting up free SSL certificate for backend..."

# You'll need a domain name pointing to your EC2 instance
echo "⚠️  This requires a domain name pointing to 54.149.156.84"
echo "Enter your domain name (e.g., api.yourdomain.com) or 'skip' to skip SSL setup:"
read DOMAIN_NAME

if [ "$DOMAIN_NAME" = "skip" ]; then
    echo "Skipping SSL setup. You can:"
    echo "1. Allow mixed content in browser (temporary)"
    echo "2. Get a domain name and run this script again"
    exit 0
fi

echo "Setting up SSL for domain: $DOMAIN_NAME"

ssh -i backend/portfolio-backend-key.pem ubuntu@54.149.156.84 << EOF
    # Install certbot
    sudo apt update
    sudo apt install -y certbot python3-certbot-nginx

    # Get SSL certificate
    sudo certbot --nginx -d $DOMAIN_NAME --non-interactive --agree-tos --email admin@$DOMAIN_NAME

    # Update backend configuration
    cd /opt/portfolio-backend
    sed -i "s|FRONTEND_URL=.*|FRONTEND_URL=https://portfolio-max-chi.vercel.app|" .env
    
    # Restart services
    sudo systemctl restart nginx
    sudo systemctl restart portfolio-backend
    
    echo "✅ SSL certificate installed!"
    echo "🌐 Your API is now available at: https://$DOMAIN_NAME"
EOF

echo "🎉 SSL setup complete!"
echo "📝 Update your frontend to use: https://$DOMAIN_NAME" 