#!/usr/bin/env bash
set -euo pipefail
mkdir -p sakila-db && cd sakila-db
curl -fsSL -o sakila-db.zip https://downloads.mysql.com/docs/sakila-db.zip
unzip -qo sakila-db.zip
mv sakila-db/sakila-schema.sql 1-sakila-schema.sql
mv sakila-db/sakila-data.sql   2-sakila-data.sql
rm -rf sakila-db sakila-db.zip
ls -la
