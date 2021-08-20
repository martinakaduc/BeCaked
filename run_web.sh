#!/bin/bash
set -o allexport
source .env
set +o allexport

while [ true ]
do
  echo "UPDATING DATA..."
  rm -rf COVID-19/csse_covid_19_data/csse_covid_19_time_series
  svn checkout --force https://github.com/CSSEGISandData/COVID-19/trunk/csse_covid_19_data/csse_covid_19_time_series COVID-19/csse_covid_19_data/csse_covid_19_time_series
  sleep 60
  echo "STARTING WEB SERVER"
  timeout 8h gunicorn -w 4 -b 0.0.0.0:$PORT "app:main()"
  sleep 8h
done
