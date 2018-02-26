#!/bin/sh -eu

echo "Waiting for postgres..."
until scan-o-matic_migrate
do
  echo "Stil waiting..."
  sleep 1
done
echo "Migration complete!"

$*
