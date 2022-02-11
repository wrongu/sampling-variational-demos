#!/bin/bash

ROOT=`pwd`

example="banana"
LAMBDAS=(1.01 1.1 1.3 1.5 2.0 4.0 8.0 16.0 32.0 64.0)
SAMPLER_ARGS="num_chains=4 num_warmup=1000 num_samples=10000"
ISVI_ARGS="num_warmup=1000 num_samples=10000 num_kl_samples=10 adapt delta=0.8"
CMDSTAN="/Users/richard/Research/libraries/cmdstan"
REFRESH=500

cd $ROOT/$example

rm *.csv *stats.txt

if [ -f data.json ]; then
    DATA_ARG="data file=data.json"
else
    DATA_ARG=""
fi

echo "ENTERED $(pwd)"
# RUN DEFAULT SAMPLING
echo "./$example sample $SAMPLER_ARGS $DATA_ARG output file=nuts.csv refresh=${REFRESH}"
./$example sample $SAMPLER_ARGS $DATA_ARG output file=nuts.csv refresh=${REFRESH}

$CMDSTAN/bin/stansummary nuts*.csv > nuts_stats.txt

# RUN DEFAULT ADVI
echo "./$example variational $DATA_ARG output file=advi.csv"
./$example variational $DATA_ARG output file=advi.csv

# RUN OURS
for l in ${LAMBDAS[@]}; do
    echo "./$example isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_1.csv refresh=${REFRESH}"
    ./$example isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_1.csv refresh=${REFRESH}

    echo "./$example isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_2.csv refresh=${REFRESH}"
    ./$example isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_2.csv refresh=${REFRESH}

    echo "./$example isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_3.csv refresh=${REFRESH}"
    ./$example isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_3.csv refresh=${REFRESH}

    echo "./$example isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_4.csv refresh=${REFRESH}"
    ./$example isvi lambda=${l} $ISVI_ARGS $DATA_ARG output file=isvi_${l}_4.csv refresh=${REFRESH}

    $CMDSTAN/bin/stansummary isvi_${l}_1.csv isvi_${l}_2.csv isvi_${l}_3.csv isvi_${l}_4.csv > isvi_${l}_stats.txt
done
cd $ROOT

# PLOTTING
mkdir figures
for l in ${LAMBDAS[@]}; do
    for c in ${CHAINS[@]}; do
        python plot_results.py --problem=$example --lam=$l --chain=$c
    done
done

