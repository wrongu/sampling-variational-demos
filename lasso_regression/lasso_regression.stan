data {
  int<lower=1> N;  // number of observations
  int<lower=1> P;  // number of features

  matrix[N, P] X;  // regressors
  vector[N] Y;     // targets
}

parameters {
  real beta0;
  vector[P] beta;
  real<lower=0> sigmaY;
}

model {
  beta0 ~ normal(0, 100);
  beta ~ double_exponential(0, 1);
  sigmaY ~ cauchy(0, 5);

  vector[N] predY = beta0 + X * beta;
  Y ~ normal(predY, sigmaY);
}