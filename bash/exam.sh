#!/bin/bash

# Assuming there are only the graphic cards mentioned in the exam
parts=("rtx3060" "rtx3070" "rtx3080" "rtx3090" "rx6700")

# variable for the otput
to_save="exam_KLEER/exam_bash/sales.txt"

# Write the date stamp
echo "$(date)" >> "$to_save"

# Loop over the parts
for part in "${parts[@]}"
do
  # Capture the result of the curl command
  result=$(curl -s "http://172.31.4.241:5000/$part")

  # Print the part and result to the sales.txt file
  echo "${part}: ${result}" >> "$to_save"	
done
