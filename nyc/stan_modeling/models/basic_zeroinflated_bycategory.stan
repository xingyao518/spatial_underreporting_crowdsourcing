// Following zero inflated modeling here: https://mc-stan.org/docs/2_28/stan-users-guide/zero-inflated.html

functions {// Using reduce sum for within-chain parallel processing. See: https://mc-stan.org/users/documentation/case-studies/reduce_sum_tutorial.html 
  real partial_sum_lpmf(int[] y_slice,
                        int start, int end,
                        matrix X_total,
                        vector logduration,
                        vector beta_total, vector theta_zeroinflation, matrix X_category) {
    // add input X_category to do zero-inflation by category
    // return poisson_log_glm_lpmf(y_slice | X_total[start:end, :], logduration[start:end], beta_total);
    int Nloc = end - start;
    real localtarget = 0;
    for (n in 1:Nloc+1) {
      int ind = start + n - 1;
      real slice_theta_zeroinflation = dot_product(X_category[ind, :], theta_zeroinflation);
      if (y_slice[n] == 0) {
        localtarget += log_sum_exp(bernoulli_lpmf(1 | slice_theta_zeroinflation),
                              bernoulli_lpmf(0 | slice_theta_zeroinflation)
                                + poisson_log_glm_lpmf(y_slice[n] | X_total[ind:ind, :], logduration[ind:ind], beta_total)
                                );
      } else {
        localtarget += bernoulli_lpmf(0 | slice_theta_zeroinflation)
                    + poisson_log_glm_lpmf(y_slice[n] | X_total[ind:ind, :], logduration[ind:ind], beta_total); 
      }
    }
    return localtarget;
  }
}

data {// Define variables in data

  // Number of observations (an integer)
  int<lower=0> N_incidents;

  int<lower=0> covariate_matrix_width;
  matrix[N_incidents, covariate_matrix_width] X; //<lower = 0, upper = 1>
  
  matrix[N_incidents, 5] X_borough; //<lower = 0, upper = 1>
  vector<lower=1,upper=1>[N_incidents] ones;


  int<lower=0> N_category;
  matrix[N_incidents, N_category] X_category; 

  // alive time for incident
  vector<lower=0>[N_incidents] duration;

  // Count outcome --- duplicates for the incident
  int<lower=0> y[N_incidents];
}

transformed data {
    vector[N_incidents] logduration;
    logduration = log(duration);

    //Combining data matrix into one
    matrix[N_incidents, 1 + 5 +covariate_matrix_width+N_category] X_total; 
    X_total = append_col(append_col(append_col(ones, X_borough), X_category), X);

}

parameters {
  // Define parameters to estimate
  vector[4] beta_borough_raw; //trick for zero centering, see: https://mc-stan.org/docs/2_28/stan-users-guide/parameterizing-centered-vectors.html
  vector[N_category-1] beta_category_raw; 

  vector[covariate_matrix_width] beta;
  real intercept;

  vector<lower=0, upper=1>[N_category] theta_zeroinflation;

}

transformed parameters  {

    vector[5] beta_borough;
    vector[N_category] beta_category;
    vector[1 + 5 + covariate_matrix_width + N_category] beta_total;

    beta_category = append_row(beta_category_raw, -sum(beta_category_raw));
    beta_borough = append_row(beta_borough_raw, -sum(beta_borough_raw));
    beta_total = append_row(append_row(append_row(intercept, beta_borough), beta_category), beta);

  }

model {
  // Prior part of Bayesian inference

    //priors for sigmas
    // sigma_category ~ normal(1, 2);
    // sigma_borough ~ normal(1, 2);
    // sigma_duration ~ normal(1, 2);
    // sigma_continuouscov ~ normal(1, 2);

    //For birth time (in days before first report), mean is 1/beta
    
    beta_category ~ normal(0, 1*inv(sqrt(1 - inv(N_category)))); //sigma_category
    beta_borough ~ normal(0, 1*inv(sqrt(1 - inv(5)))); //sigma_borough
    beta ~ normal(0, 1); //sigma_continuouscov
    intercept ~ normal(0, 5);

    // Ok, the final reduce sum thing
    int grainsize = 1;
    target += reduce_sum(
      partial_sum_lpmf, y, grainsize, X_total, logduration, beta_total, theta_zeroinflation, X_category
      );

}



generated quantities {
  //For posterior predictive check https://mc-stan.org/docs/2_28/stan-users-guide/simulating-from-the-posterior-predictive-distribution.html
  //https://mc-stan.org/docs/2_19/functions-reference/poisson-distribution-log-parameterization.html
    array[N_incidents] int y_rep;
    vector[N_incidents] log_likelihood;
    for (i in 1:N_incidents) {
      int intsamplezero = bernoulli_rng(dot_product(X_category[i], theta_zeroinflation));
      y_rep[i] = (1 - intsamplezero)*poisson_log_rng(X_total[i]*beta_total + logduration[i]);

      if (y[i] == 0) {
          log_likelihood[i]= log_sum_exp(bernoulli_lpmf(1 | theta_zeroinflation),
                                bernoulli_lpmf(0 | theta_zeroinflation)
                                  + poisson_log_lpmf(y[i] | X_total[i]*beta_total + logduration[i])
                                  );
        } else {
          log_likelihood[i] = bernoulli_lpmf(0 | theta_zeroinflation)
                      + poisson_log_lpmf(y[i] | X_total[i]*beta_total + logduration[i]);
        }
    } 
}