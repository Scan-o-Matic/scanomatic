#!/bin/sh -eux

user=scanomatic
db=scanomatic

echo "[$BASH_SOURCE] ⚠ THIS SCRIPT CREATES A POSTGRES USER WITHOUT A PASSWORD. DO NOT USE IN PRODUCTION ⚠" >&2
echo "[$BASH_SOURCE] Creating database user $user"
createuser $user

echo "[$BASH_SOURCE] Creating database $db"
createdb $db

echo "[$BASH_SOURCE] Configuring databse $db"
psql -c "grant all privileges on database $db to $user ;"
psql --dbname $db -c "CREATE EXTENSION btree_gist;"
