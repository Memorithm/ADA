#!/usr/bin/env bash
set -euo pipefail

failed=0
while IFS= read -r entry; do
  if [[ "$entry" =~ uses:[[:space:]]*([^[:space:]#]+) ]]; then
    reference="${BASH_REMATCH[1]}"
    case "$reference" in
      ./*|docker://*) continue ;;
    esac
    if [[ ! "$reference" =~ @[0-9a-f]{40}$ ]]; then
      echo "Unpinned external action: $entry" >&2
      failed=1
    fi
  fi
done < <(
  grep -REn --include='*.yml' --include='*.yaml' \
    'uses:[[:space:]]*[^[:space:]#]+' .github/workflows || true
)

exit "$failed"
