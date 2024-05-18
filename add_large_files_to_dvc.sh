#!/bin/bash

# Define size threshold
size_threshold=${1:-45} # Default to 45 MB if no argument provided
size_threshold=$((size_threshold * 1024 * 1024)) # Convert MB to bytes

# Ensure clean working directory
if ! git diff --quiet || ! git diff --staged --quiet; then
    echo "Please commit or stash your changes before running this script."
    exit 1
fi

# Initialize DVC if not already done
if [ ! -d ".dvc" ]; then
    dvc init
    git commit -m "Initialize DVC"
fi

# Find and process large files
find . -type f -size +${size_threshold}c ! -path "./.git/*" ! -path "./.dvc/*" ! -path "./.gitignore" -print0 | while read -r -d $'\0' file; do
    relative_path=${file#./} # Remove the leading "./" for relative paths

    # Remove file from Git tracking if it's tracked
    if git ls-files --error-unmatch "$relative_path" > /dev/null 2>&1; then
        echo "Removing $relative_path from Git tracking..."
        git rm --cached "$relative_path" --quiet
    fi

    # Ensure .gitignore is updated to ignore the file globally
    echo "$relative_path" >> .gitignore

    echo "Adding $relative_path to DVC..."
    dvc add "$relative_path" --quiet

    echo "$relative_path added to DVC and .gitignore."
done

# Commit the updated .gitignore and .dvc files
git add .gitignore *.dvc
git commit -m "Update DVC tracking and .gitignore"

# Push changes
git push origin main
dvc push
