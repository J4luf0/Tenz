#!/bin/bash

INPUT="test/src/core/Tensor_stress.cpp"
OUTPUT="test/src/core/TensorParallel_stress.cpp"

declare -A REPLACEMENTS=(
  ["Tensor"]="TensorParallel"
  ["tensor_stress_test"]="tensorparallel_stress_test"
)

cmd="sed"

for key in "${!REPLACEMENTS[@]}"; do
  cmd+=" -e s|${key}|${REPLACEMENTS[$key]}|g"
done

$cmd "$INPUT" > "$OUTPUT"
