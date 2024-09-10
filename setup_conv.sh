#!/bin/bash

# Set the base directory to the extensions folder
BASE_DIR="extensions"

# Check if the directory exists
if [ ! -d "$BASE_DIR" ]; then
  echo "The directory $BASE_DIR does not exist."
  exit 1
fi

# Iterate over each subdirectory in the base directory
for dir in "$BASE_DIR"/*/
do
  # Check if it's a directory
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"
    
    # Change to the subdirectory
    cd "$dir" || { echo "Failed to switch to directory $dir"; exit 1; }
    
    # Check if setup.py exists
    if [ -f "setup.py" ]; then
      # Run the python setup.py install command
      python setup.py install
    else
      echo "No setup.py found in directory $dir"
    fi
    
    # Return to the original directory
    cd - > /dev/null
  else
    echo "$dir is not a directory"
  fi
done

echo "Processing complete."

# PS：此脚本用于切换到extensions下的每个子目录下并执行 python setup.py install