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

echo "Fetching caches older than: $CUTOFF..."

# Fetch cache metadata (ID, key, and size in bytes) to provide a rich summary later
CACHE_DATA=$(gh cache list -R "$REPO" --limit 100 --json id,key,sizeInBytes,createdAt | \
jq -c --arg cutoff "$CUTOFF" '.[] | select(.createdAt < $cutoff)')

if [ -z "$CACHE_DATA" ]; then
    echo "No old caches found to delete."
    exit 0
fi

# Convert JSON lines to a Bash array
mapfile -t CACHE_ITEMS < <(echo "$CACHE_DATA")
TOTAL_CACHES=${#CACHE_ITEMS[@]}

echo "Found $TOTAL_CACHES caches older than 24 hours."

# Tracking variables
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_BYTES_FREED=0

CURRENT=0
for item in "${CACHE_ITEMS[@]}"; do
    ((CURRENT++))
    
    # Extract properties using jq
    id=$(echo "$item" | jq -r '.id')
    key=$(echo "$item" | jq -r '.key')
    size=$(echo "$item" | jq -r '.sizeInBytes')
    
    echo "[$CURRENT/$TOTAL_CACHES] Deleting ID: $id (Key: $key)..."
    
    # Attempt deletion and check exit status
    if gh cache delete "$id" -R "$REPO" 2>/dev/null; then
        ((SUCCESS_COUNT++))
        ((TOTAL_BYTES_FREED+=size))
    else
        echo "  --> [ERROR] Failed to delete cache ID: $id"
        ((FAIL_COUNT++))
    fi
done

# Convert total bytes freed into a human-readable format
if [ "$TOTAL_BYTES_FREED" -ge 1073741824 ]; then
    SIZE_HUMAN=$(printf "%.2f GB" "$(echo "scale=2; $TOTAL_BYTES_FREED / 1073741824" | bc)")
elif [ "$TOTAL_BYTES_FREED" -ge 1048576 ]; then
    SIZE_HUMAN=$(printf "%.2f MB" "$(echo "scale=2; $TOTAL_BYTES_FREED / 1048576" | bc)")
else
    SIZE_HUMAN=$(printf "%.2f KB" "$(echo "scale=2; $TOTAL_BYTES_FREED / 1024" | bc)")
fi

# Detailed final summary report
echo "=================================================="
echo "CLEANUP PROCESS COMPLETE"
echo "=================================================="
echo "Repository:      $REPO"
echo "Cutoff Time:     $CUTOFF"
echo "Total Target:    $TOTAL_CACHES caches found"
echo "Successfully:    $SUCCESS_COUNT deleted"
echo "Failed:          $FAIL_COUNT errors encountered"
echo "Storage Freed:   $SIZE_HUMAN"
echo "=================================================="
