#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y nginx python3-venv python3-pip ca-certificates curl

install -d -o root -g root /srv/who-has-shade
install -d -o www-data -g www-data /srv/who-has-shade/report
install -d -o www-data -g www-data /srv/who-has-shade/atlas
install -d -o www-data -g www-data /srv/who-has-shade/api
install -d -o www-data -g www-data /srv/who-has-shade/data/cogs
install -d -o www-data -g www-data /var/cache/nginx/canopy
install -d -o www-data -g www-data /var/cache/nginx/basemap

if ! swapon --show | grep -q .; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

systemctl enable nginx
