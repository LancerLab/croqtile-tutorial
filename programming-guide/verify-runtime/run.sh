#!/bin/sh
# Build and run every harness against the locally built Choreo compiler and
# the GCU device.  Usage: ./run.sh [path-to-choreo]
CHOREO=${1:-/home/wsj/choreo/build/choreo}
INC=/home/wsj/choreo/runtime
here=$(dirname "$0")
mkdir -p "$here/out"
fail=0
for f in "$here"/harnesses/*.co; do
  n=$(basename "$f" .co)
  if ! "$CHOREO" -t topscc -arch=gcu300 -I"$INC" "$f" -o "$here/out/$n" >"$here/out/$n.log" 2>&1; then
    echo "COMPILE-FAIL $n"; sed -n '1,5p' "$here/out/$n.log"; fail=1; continue
  fi
  if out=$("$here/out/$n" 2>&1); then
    echo "PASS $n : $out"
  else
    echo "FAIL $n : $out"; fail=1
  fi
done
exit $fail
