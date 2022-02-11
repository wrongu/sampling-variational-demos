data {}

parameters {
  real x;
  real y;
}

model {
  target += (-x*x*x*x/8 + y*x*x - 2*y*y - x*x/4);
}