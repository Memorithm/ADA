#!/usr/bin/env bash
set -euo pipefail

mapfile -d '' tracked_files < <(git ls-files -z)
scan_files=()
for file in "${tracked_files[@]}"; do
  case "$file" in
    .github/workflows/*.yml|.github/workflows/*.yaml|action.yml|action.yaml|*/action.yml|*/action.yaml)
      scan_files+=("$file")
      ;;
  esac
done

if (( ${#scan_files[@]} == 0 )); then
  echo "No workflow or action manifests found" >&2
  exit 1
fi

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
done < <(grep -nH -E 'uses:[[:space:]]*[^[:space:]#]+' "${scan_files[@]}" || true)

exit "$failed"
