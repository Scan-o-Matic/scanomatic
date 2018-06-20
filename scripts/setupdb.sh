#!/bin/sh -eux

user=scanomatic
db=scanomatic

name=$0

echo "[$name] ⚠ THIS SCRIPT CREATES A POSTGRES USER WITHOUT A PASSWORD. DO NOT USE IN PRODUCTION ⚠" >&2
echo "[$name] Creating database user $user"
createuser $user

echo "[$name] Creating database $db"
createdb $db

echo "[$name] Configuring databse $db"
psql -c "grant all privileges on database $db to $user ;"
psql --dbname $db -c "CREATE EXTENSION btree_gist;"
