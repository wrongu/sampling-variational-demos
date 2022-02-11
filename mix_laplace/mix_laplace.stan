data {
  vector[2] means;
  vector[2] scales;
  real weight;
}

parameters {
  real x;
}

model {
  real log_mode_1 = -fabs(x - means[1]) / scales[1] - log(2*scales[1]);
  real log_mode_2 = -fabs(x - means[2]) / scales[2] - log(2*scales[2]);
  target += log_sum_exp(log_mode_1 + log(weight), log_mode_2 + log(1-weight));
}