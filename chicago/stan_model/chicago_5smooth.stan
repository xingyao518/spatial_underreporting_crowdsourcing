functions {// Using reduce sum for within-chain parallel processing. See: https://mc-stan.org/users/documentation/case-studies/reduce_sum_tutorial.html 
  real partial_sum_lpmf(int[] y_slice,
                        int start, int end,
                        matrix X_total,
                        vector logduration,
                        vector beta_total, real theta_zeroinflation) {
    //have to edit into for loop because of doing the zero inflation
    // return poisson_log_glm_lpmf(y_slice | X_total[start:end, :], logduration[start:end], beta_total);
    int Nloc = end - start;
    real localtarget = 0;
    for (n in 1:Nloc+1) {
      int ind = start + n - 1; //not sure if this is correct
        if (y_slice[n] == 0) {
          localtarget += log_sum_exp(bernoulli_lpmf(1 | theta_zeroinflation),
                                bernoulli_lpmf(0 | theta_zeroinflation)
                                  + poisson_log_glm_lpmf(y_slice[n] | X_total[ind:ind, :], logduration[ind:ind], beta_total)
                                  );
        } else {
          localtarget += bernoulli_lpmf(0 | theta_zeroinflation)
                      + poisson_log_glm_lpmf(y_slice[n] | X_total[ind:ind, :], logduration[ind:ind], beta_total); 
        }
    }
    return localtarget;
  }
}

data {// Define variables in data

  // Number of observations (an integer)
  int<lower=0> N_incidents;

  // int<lower=0> covariate_matrix_width;
  // matrix[N_incidents, covariate_matrix_width] X; //<lower = 0, upper = 1>
  
  vector<lower=1,upper=1>[N_incidents] ones;


  int<lower=0> N_category;
  int<lower=0> N_tract;

  int<lower=0> N_edges; //tract adjacency matrix number of edges //Code from https://mc-stan.org/workshops/dec2017/spatial_smoothing_icar.html#8
  int<lower=1, upper=N_tract> node1[N_edges];  // node1[i] adjacent to node2[i]
  int<lower=1, upper=N_tract> node2[N_edges];  // and node1[i] < node2[i]

  matrix[N_incidents, N_category] X_category;

  matrix[N_incidents, N_tract] X_tract; 

  // alive time for incident
  vector<lower=0>[N_incidents] duration;

  // Count outcome --- duplicates for the incident
  int<lower=0> y[N_incidents];
}

transformed data {
    vector[N_incidents] logduration;
    logduration = log(duration);

    matrix[N_incidents, 1 + N_tract + N_category] X_total; 
    X_total = append_col(append_col(ones, X_tract), X_category);

}

parameters {
  // Define parameters to estimate
  vector[N_tract - 1] beta_tract_raw; //trick for zero centering, see: https://mc-stan.org/docs/2_28/stan-users-guide/parameterizing-centered-vectors.html
  vector[N_category -1] beta_category_raw; 

  // vector[covariate_matrix_width] beta; 
  real intercept;
  // real<lower=0> sigma_tract;

  real<lower=0, upper=1> theta_zeroinflation;
}

transformed parameters  {

    vector[N_tract] beta_tract;
    vector[N_category ] beta_category;
    vector[1 + N_tract + N_category] beta_total;

    beta_category = append_row(beta_category_raw, -sum(beta_category_raw));

    // beta_tract = append_row(beta_tract_raw, -sum(beta_tract_raw)); //trying new way to do this besides append row
    beta_tract[1:(N_tract - 1)] = beta_tract_raw;
    beta_tract[N_tract] = -sum(beta_tract_raw);
        
    beta_total = append_row(append_row(intercept, beta_tract), beta_category);

  }

model {
  // Prior part of Bayesian inference
    // sigma_tract ~ normal(1, 3);   
    beta_tract ~ normal(0, 1); 
    beta_category ~ normal(0, 2*inv(sqrt(1 - inv(N_category)))); //sigma_category
    // beta ~ normal(0, 1); //sigma_continuouscov
    intercept ~ normal(0, 5);

    // Ok, the final reduce sum thing
    int grainsize = 1;
    target += reduce_sum(
      partial_sum_lpmf, y, grainsize, X_total, logduration, beta_total, theta_zeroinflation
      );

    // add adjacency priors to beta_tract
    target += -5 * dot_self(beta_tract[node1] - beta_tract[node2]);
}



generated quantities {
  //For posterior predictive check https://mc-stan.org/docs/2_28/stan-users-guide/simulating-from-the-posterior-predictive-distribution.html
  //https://mc-stan.org/docs/2_19/functions-reference/poisson-distribution-log-parameterization.html
    array[N_incidents] int y_rep;
    vector[N_incidents] log_likelihood;
    for (i in 1:N_incidents) {
      int intsamplezero = bernoulli_rng(theta_zeroinflation);
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