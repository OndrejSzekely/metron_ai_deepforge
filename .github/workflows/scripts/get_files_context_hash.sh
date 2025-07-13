#! /usr/bin/bash

get_files_context_hash() {
    local FILES="$1"
    local EXISTING_FILES=""
    
    for f in $FILES; do
        if [ -e "$f" ]; then
            EXISTING_FILES="$EXISTING_FILES $f"
        fi
    done

    if [ -z "$EXISTING_FILES" ]; then
        echo "ERROR: No valid files found in input." >&2
        return 1
    fi

    tar --sort=name --mtime=0 --owner=0 --group=0 --numeric-owner -cf - $EXISTING_FILES 2>/dev/null | sha256sum | cut -d ' ' -f1
}