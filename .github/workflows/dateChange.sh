#!/bin/bash
# Path to the file to edit
DEFAULT_FILE="/workspace/CIRRUS-MILES-CREDIT/model_predict_old.yml"

FILE="${1:-$DEFAULT_FILE}"

if [[ ! -f "$FILE" ]]; then
  echo "Error: file not found: $FILE" >&2
  exit 1
else
  echo "Changing date on file: $FILE"
fi

#FILE="/__w/CIRRUS-MILES-CREDIT/.github/workflows/model_predict_CI.yml"

# Determine which 6-hour window we're in
#hour=$(date +%H)
TZ="America/Denver"
hour=$(TZ="$TZ" date +%H)
case $hour in
  00|01|02|03|04|05) start_hour=00 ;;
  06|07|08|09|10|11) start_hour=06 ;;
  12|13|14|15|16|17) start_hour=12 ;;
  *) start_hour=18 ;;
esac

date_str=$(TZ="$TZ" date +%Y-%m-%d)
start_time="${date_str} $(printf "%02d" $start_hour):00:00"

# Get epoch time for now in Denver
epoch_start=$(date --date="$start_time" +%s)

# Add 6 hours (in seconds)
epoch_end=$((epoch_start + 6*3600))

# Format both in Denver local time
#start_time=$(TZ="America/Denver" date -d @"$epoch_start" +"%Y-%m-%d %H:%M:%S")
start_time=$(TZ="$TZ" date -d @"$epoch_start" +"%Y-%m-%d %H:%M:%S")
#end_time=$(TZ="America/Denver" date -d @"$epoch_end" +"%Y-%m-%d %H:%M:%S")
end_time=$(TZ="$TZ" date -d @"$epoch_end" +"%Y-%m-%d %H:%M:%S")
echo "start=$start_time end=$end_time"

sed -i \
  -e "s|^\(.*forecast_start_time: *\).*|\1\"${start_time}\"|" \
  -e "s|^\(.*forecast_end_time: *\).*|\1\"${end_time}\"|" \
  "$FILE"
