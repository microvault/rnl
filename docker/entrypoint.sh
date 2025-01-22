!/bin/bash
set -e

apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0

apt-get clean && rm -rf /var/lib/apt/lists/*

echo ""
echo "Starting training ..."
echo ""

exec python -m main learn
