#!/bin/bash

# Arrays to store package paths and tracking
declare -a packages=("$@")
declare -a failed_packages=()
declare -a permanent_failures=()
# Replace associative array with simple naming convention for retry counts
declare -a retry_counts=()
made_progress=true

# Function to get retry count for a package
get_retry_count() {
    local package="$1"
    local index=0
    for p in "${packages[@]}"; do
        if [ "$p" = "$package" ]; then
            echo "${retry_counts[$index]:-0}"
            return
        fi
        ((index++))
    done
    echo "0"
}

# Function to set retry count for a package
set_retry_count() {
    local package="$1"
    local count="$2"
    local index=0
    for p in "${packages[@]}"; do
        if [ "$p" = "$package" ]; then
            retry_counts[$index]=$count
            return
        fi
        ((index++))
    done
}

# Function to attempt locking and publishing a package
publish_package() {
    local package_path="$1"
    echo "Processing package: $package_path"

    cd "$package_path" || return 1

    if poetry lock; then
        if poetry publish --build; then
            echo "Successfully published $package_path"
            cd - > /dev/null
            return 0
        fi
    fi

    cd - > /dev/null
    return 1
}

# Main loop - continue as long as we're making progress
# Main loop - continue as long as we're making progress
while [ ${#packages[@]} -gt 0 ] && $made_progress; do
    made_progress=false
    failed_packages=()

    echo "Starting new publishing pass with ${#packages[@]} packages"

    # Try to publish each remaining package
    for package in "${packages[@]}"; do
        # Get and increment retry count
        current_count=$(get_retry_count "$package")
        new_count=$((current_count + 1))
        set_retry_count "$package" "$new_count"

        # Check if we've exceeded retry limit
        if [ "$new_count" -gt 3 ]; then
            echo "Package $package has failed 3 times, will not retry"
            permanent_failures+=("$package")
            continue
        fi

        if publish_package "$package"; then
            made_progress=true
        else
            echo "Failed to publish $package (attempt ${new_count})"
            failed_packages+=("$package")
        fi
    done

    # Update packages array with failed ones for next iteration
    packages=("${failed_packages[@]}")

    echo "Pass completed. ${#failed_packages[@]} packages remaining"

    # sleep for 60 seconds
    sleep 60
done

# Print final status
echo
echo "=== Publishing Summary ==="
if [ ${#packages[@]} -eq 0 ] && [ ${#permanent_failures[@]} -eq 0 ]; then
    echo "All packages published successfully! 🎉"
    exit 0
else
    if [ ${#permanent_failures[@]} -gt 0 ]; then
        echo "Packages that failed after 3 attempts:"
        printf '%s\n' "${permanent_failures[@]}"
    fi
    if [ ${#packages[@]} -gt 0 ]; then
        echo "Packages that failed due to dependency resolution:"
        printf '%s\n' "${packages[@]}"
    fi
    exit 1
fi
