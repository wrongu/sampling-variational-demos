#!/bin/bash

ROOT=`pwd`

PROBLEM="$1"
LAMBDAS=($(python3 -c "import numpy; print(*map(lambda x: f'{x:.3f}', numpy.logspace(0,1,21)))"))
SAMPLER_ARGS="num_warmup=1000 num_samples=10000"
ISVI_ARGS="num_warmup=1000 stochastic_kl=1 num_samples=10000 num_kl_samples=50 adapt delta=0.8"
CMDSTAN="$ROOT/cmdstan"
REFRESH=500
CHAINS=(1 2 3 4)

if [ ! -f $ROOT/$PROBLEM/$PROBLEM ]; then
    echo "$PROBLEM/$PROBLEM does not exist - maybe needs to be compiled?"
    exit 1
fi

cd $ROOT/$PROBLEM

if [ -f data.json ]; then
    DATA_ARG="data file=data.json"
else
    DATA_ARG=""
fi

COMMANDS=()

for c in ${CHAINS[@]}; do
    # RUN NUTS
    if [ ! -e nuts_${c}.csv ]; then
        echo "Running nuts_${c}.csv"
	COMMANDS+=("./$PROBLEM sample $SAMPLER_ARGS $DATA_ARG output file=nuts_${c}.csv refresh=${REFRESH}")
    else
        echo "Skipping nuts_${c}.csv"
    fi

    # RUN ADVI
    if [ ! -e advi_${c}.csv ]; then
        echo "Running advi_${c}.csv"
	COMMANDS+=("./$PROBLEM variational $DATA_ARG output file=advi_${c}.csv > /dev/null")
    else
        echo "Skipping advi_${c}.csv"
    fi
done

# RUN OURS (ISVI), FOR EACH VALUE OF LAMBDA
for l in ${LAMBDAS[@]}; do
    for c in ${CHAINS[@]}; do
        if [ ! -e isvi_${l}_${c}.csv ]; then
            echo "Running isvi_${l}_${c}.csv"
	    COMMANDS+=("./$PROBLEM isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_${c}.csv refresh=${REFRESH}")
        else
            echo "Skipping isvi_${l}_${c}.csv"
        fi
    done
done

# COMMANDS COLLECTED - NOW ACTUALLY DO THE RUNNING
parallel --jobs=12 --linebuffer ::: "${COMMANDS[@]}"

# Summarize stats for NUTS and ISVI
$CMDSTAN/bin/stansummary nuts*.csv > nuts_stats.txt
for l in ${LAMBDAS[@]}; do
    # Summarize stats for ISVI across all chains
    $CMDSTAN/bin/stansummary isvi_${l}_*.csv > isvi_${l}_stats.txt
done
