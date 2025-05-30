// Following zero inflated modeling here: https://mc-stan.org/docs/2_28/stan-users-guide/zero-inflated.html

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

  int<lower=0> covariate_matrix_width;
  matrix[N_incidents, covariate_matrix_width] X; //<lower = 0, upper = 1>
  
  matrix[N_incidents, 5] X_borough; //<lower = 0, upper = 1>
  vector<lower=1,upper=1>[N_incidents] ones;


  int<lower=0> N_risk;
  matrix[N_incidents, N_risk] X_risk; 

  int<lower=0> N_category;
  matrix[N_incidents, N_category] X_category;

  int<lower=0> N_borough_riskbin; // number of interaction terms
  matrix[N_incidents, N_borough_riskbin] X_borough_riskbin; // interaction terms design matrix

  // alive time for incident
  vector<lower=0>[N_incidents] duration;

  // Count outcome --- duplicates for the incident
  int<lower=0> y[N_incidents];
}

transformed data {
    vector[N_incidents] logduration;
    logduration = log(duration);

    //Combining data matrix into one
    matrix[N_incidents, 1 + 5 +covariate_matrix_width+N_risk+N_borough_riskbin+N_category] X_total; 
    X_total = append_col(append_col(append_col(append_col(append_col(ones, X_borough_riskbin), X_borough), X_risk), X_category),X);


}

parameters {
  // Define parameters to estimate
  vector[4] beta_borough_raw; //trick for zero centering, see: https://mc-stan.org/docs/2_28/stan-users-guide/parameterizing-centered-vectors.html
  vector[N_risk-1] beta_risk_raw; 
  vector[N_category-1] beta_category_raw;
  matrix[4, N_risk-1] beta_borough_riskbin_raw;

  vector[covariate_matrix_width] beta;
  real intercept;

  real<lower=0, upper=1> theta_zeroinflation;


}

transformed parameters  {

    vector[5] beta_borough;
    vector[N_risk] beta_risk;
    vector[N_category] beta_category;
    vector[N_borough_riskbin] beta_borough_riskbin;
    vector[1 + 5 + covariate_matrix_width + N_risk + N_borough_riskbin] beta_total;

    beta_risk = append_row(beta_risk_raw, -sum(beta_risk_raw));
    beta_borough = append_row(beta_borough_raw, -sum(beta_borough_raw));
    beta_category = append_row(beta_category_raw, -sum(beta_category_raw));
    
    
    beta_borough_riskbin = to_vector(beta_borough_riskbin_raw); // not sure this is correct
    

    beta_total = append_row(append_row(append_row(append_row(append_row(intercept, beta_borough_riskbin), beta_borough), beta_risk), beta_category), beta);

  }

model {
  // Prior part of Bayesian inference
    
    beta_risk ~ normal(0, 1*inv(sqrt(1 - inv(N_risk)))); //sigma_category
    beta_borough ~ normal(0, 1*inv(sqrt(1 - inv(5)))); //sigma_borough
    beta_borough_riskbin ~ normal(0, 1*inv(sqrt(1 - inv(N_borough_riskbin)))); //sigma_borough
    beta_category ~ normal(0, 1*inv(sqrt(1 - inv(N_category)))); //sigma_category
    beta ~ normal(0, 1); //sigma_continuouscov
    intercept ~ normal(0, 5);

    // for (i in 1:(N_category-1)) {
    //   sum(beta_borough_category_raw[i,]) ~ normal(0, 0.01); //sigma_borough
    // }
    // for (i in 1:4) {
    //   sum(beta_borough_category_raw[,i]) ~ normal(0, 0.01); //sigma_borough
    // }


    // Ok, the final reduce sum thing
    int grainsize = 1;
    target += reduce_sum(
      partial_sum_lpmf, y, grainsize, X_total, logduration, beta_total, theta_zeroinflation
      );

}



generated quantities {
  //For posterior predictive check https://mc-stan.org/docs/2_28/stan-users-guide/simulating-from-the-posterior-predictive-distribution.html
  //https://mc-stan.org/docs/2_19/functions-reference/poisson-distribution-log-parameterization.html
    // array[N_incidents] int y_rep;
    // vector[N_incidents] log_likelihood;
    // for (i in 1:N_incidents) {
    //   int intsamplezero = bernoulli_rng(theta_zeroinflation);
    //   y_rep[i] = (1 - intsamplezero)*poisson_log_rng(X_total[i]*beta_total + logduration[i]);

    //   if (y[i] == 0) {
    //       log_likelihood[i]= log_sum_exp(bernoulli_lpmf(1 | theta_zeroinflation),
    //                             bernoulli_lpmf(0 | theta_zeroinflation)
    //                               + poisson_log_lpmf(y[i] | X_total[i]*beta_total + logduration[i])
    //                               );
    //     } else {
    //       log_likelihood[i] = bernoulli_lpmf(0 | theta_zeroinflation)
    //                   + poisson_log_lpmf(y[i] | X_total[i]*beta_total + logduration[i]);
    //     }
    // } 

    // //For hazard delay
    // vector[N_hazard] delay;
    // for (i in 1:N_hazard) {
    //   delay[i] = exponential_rng(exp(X_total[ind_hazard[i],:]*beta_total));
    // }
    vector[N_incidents] delay;
    for (i in 1:N_incidents) {
      delay[i] = exponential_rng(exp(X_total[i,:]*beta_total));
    }
}