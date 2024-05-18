#!/bin/bash

# Ensure the script is executed in the root directory of a git repository
if [ ! -d ".git" ]; then
    echo "Error: Must be run at the root of a Git repository."
    exit 1
fi

# Check if git filter-repo is installed
if ! command -v git-filter-repo &> /dev/null
then
    echo "git-filter-repo could not be found, please install it first."
    exit 1
fi

# Confirm before proceeding
echo "This script will permanently remove files from your repository history based on .gitignore, without deleting them from your working directory."
read -p "Are you sure you want to proceed? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted by user."
    exit 1
fi

# Backup the current .gitignore
cp .gitignore .gitignore.backup

# Temporarily move matched files to a safe location
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

# Read .gitignore and filter each entry
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    if [[ "$line" == \#* ]] || [[ -z "$line" ]]; then
        continue
    fi
    
    # Prepare the path for filter-repo
    pattern="${line/#\//}"  # Remove leading slash if present for consistency
    echo "Processing $pattern..."

    # Move files to a temporary directory
    find . -path "$pattern" -exec mv {} "$tmp_dir" \;

    # Run git-filter-repo to remove files from history
    git filter-repo --path "$pattern" --invert-paths --force
    
    # Move files back from the temporary directory
    find "$tmp_dir" -exec mv {} . \;
done < .gitignore

# Commit the changes if any .gitignore was modified
git add .gitignore *.dvc
git commit -m "Update DVC tracking and .gitignore after history rewrite"

# Force push the changes
echo "History rewrite complete. Please verify the integrity of the repository before pushing."
echo "If everything is correct, use 'git push --force' to update the remote repository."
