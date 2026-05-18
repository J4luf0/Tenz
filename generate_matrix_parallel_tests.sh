#!/bin/bash

INPUT="test/src/core/Matrix_test.cpp"
OUTPUT="test/src/core/MatrixParallel_test.cpp"

declare -A REPLACEMENTS=(
  ["Matrix"]="MatrixParallel"
  ["matrix_test"]="matrixparallel_test"
)

cmd="sed"

for key in "${!REPLACEMENTS[@]}"; do
  cmd+=" -e s|${key}|${REPLACEMENTS[$key]}|g"
done

$cmd "$INPUT" > "$OUTPUT"
