#!/bin/bash
set -e

# Create directories
mkdir -p /var/lib/postgresql/data
mkdir -p /var/lib/postgresql/certs

# Generate self-signed certificate and key
openssl req -new -x509 -days 365 -nodes \
    -text -out /var/lib/postgresql/certs/server.crt \
    -keyout /var/lib/postgresql/certs/server.key \
    -subj "/CN=gira-postgres"

# Set proper permissions
chmod 600 /var/lib/postgresql/certs/server.key
chown postgres:postgres /var/lib/postgresql/certs/server.key /var/lib/postgresql/certs/server.crt
