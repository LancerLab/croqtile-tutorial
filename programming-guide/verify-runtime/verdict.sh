#!/bin/sh
CHOREO=${1:-/home/wsj/choreo/build/choreo}
here=$(dirname "$0")
for f in "$here"/blocks/*.co; do
  n=$(basename "$f" .co)
  out=$("$CHOREO" -t topscc -arch=gcu300 -es "$f" 2>&1); rc=$?
  if [ $rc -ge 128 ]; then
    printf '%-12s | CRASH(%s) | %s\n' "$n" "$rc" "$(echo "$out" | grep -E 'Assertion|Segmentation' | head -1 | sed 's/.*Assertion/Assertion/' | cut -c1-90)"
  elif echo "$out" | grep -q "error:"; then
    printf '%-12s | FAIL | %s\n' "$n" "$(echo "$out" | grep -m1 'error:' | sed 's/.*error: //' | cut -c1-95)"
  else
    printf '%-12s | ok   |\n' "$n"
  fi
done
