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

for c in ${CHAINS[@]}; do
    # RUN NUTS
    if [ ! -f nuts_${c}.csv ]; then
        ./$PROBLEM sample $SAMPLER_ARGS $DATA_ARG output file=nuts_${c}.csv refresh=${REFRESH}
    fi

    # RUN ADVI
    if [ ! -f advi_${c}.csv ]; then
        ./$PROBLEM variational $DATA_ARG output file=advi_${c}.csv
    fi
done

# Summarize stats for NUTS across all chains
$CMDSTAN/bin/stansummary nuts*.csv > nuts_stats.txt

# RUN OURS (ISVI), FOR EACH VALUE OF LAMBDA
for l in ${LAMBDAS[@]}; do
    for c in ${CHAINS[@]}; do
        if [ ! -f isvi_${l}_${c}.csv ]; then
            ./$PROBLEM isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_${c}.csv refresh=${REFRESH}
        fi
    done

    # Summarize stats for ISVI across all chains
    $CMDSTAN/bin/stansummary isvi_${l}_*.csv > isvi_${l}_stats.txt
done
