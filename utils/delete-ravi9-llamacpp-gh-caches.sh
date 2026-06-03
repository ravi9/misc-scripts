#!/bin/bash
# Script to delete https://github.com/ravi9/llama.cpp/actions/caches caches older than 1 day

# Target repository
REPO="ravi9/llama.cpp"

# Calculate the cutoff timestamp for 24 hours ago (ISO 8601 UTC)
if date --version >/dev/null 2>&1; then
    # GNU date (Linux)
    CUTOFF=$(date -u -d "1 day ago" +"%Y-%m-%dT%H:%M:%SZ")
else
    # BSD date (macOS)
    CUTOFF=$(date -u -v-1d +"%Y-%m-%dT%H:%M:%SZ")
fi

echo "Deleting caches created before: $CUTOFF"

# 1. Fetch all caches from the repo in JSON format
# 2. Filter for caches where 'createdAt' is less than the cutoff
# 3. Extract their IDs and delete them one by one
gh cache list -R "$REPO" --limit 100 --json id,createdAt | \
jq -r --arg cutoff "$CUTOFF" '.[] | select(.createdAt < $cutoff) | .id' | \
while read -r cache_id; do
    if [ -n "$cache_id" ]; then
        echo "Deleting cache ID: $cache_id"
        gh cache delete "$cache_id" -R "$REPO"
    fi
done

echo "Cleanup complete!"
