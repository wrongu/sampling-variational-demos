data {
  real<lower=-1, upper=+1> corr;
}

parameters {
  vector[2] x;
}

model {
  vector[2] mu = [0, 0]';
  matrix[2,2] cov = [[1, corr], [corr, 1]];
  x ~ multi_normal(mu, cov);
}