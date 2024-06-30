#!/bin/bash
# Generate annotations for all records in the databases.

RECORDS_SET_P=(databases/set-p/*.hea)
RECORDS_TRAINING=(databases/training/*.hea)

# Directories for annotations
rm -rf annotations/
mkdir -p annotations/set-p annotations/training

# Eval set-p
for record in "${RECORDS_SET_P[@]}"; do
    python detector.py "${record%.hea}" > /dev/null
    mv "$(basename "$record" .hea).asc" annotations/set-p/
done

# Eval training
for record in "${RECORDS_TRAINING[@]}"; do
    python detector.py "${record%.hea}" > /dev/null
    mv "$(basename "$record" .hea).asc" annotations/training/
done
