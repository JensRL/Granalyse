// relation_model.stan
// ===================
// Hierarchical Stan model for deriving granulation-radius scaling relations.
//
// Regression (all quantities in relative log space, normalised to the reference star):
//   log10(R_rel) = fac + x*log10(A_gran_rel) + y*log10(charb_rel)
//                      + z*log10(Teff_rel) + z2*(feh - feh_ref)
//
// This is the Gaussian measurement-error variant used for both shipped
// relation sets (ms_sgb and rgb).
//

data {
    int<lower = 0> N;
    vector[N] lg_amp_obs;            // observed log10(A_gran) [relative to reference star]
    vector[N] lg_charb_obs;          // observed log10(charb)  [relative to reference star]
    array[N] matrix[2, 2] gran_cov;  // 2x2 covariance for each star [lg_amp, lg_charb]
    vector[N] lg_teff_obs;           // observed log10(Teff)   [relative to reference star]
    vector<lower=0>[N] lg_teff_err;  // uncertainty in log10(Teff)
    vector[N] lg_R_obs;              // observed log10(R)       [relative to reference star]
    vector<lower=0>[N] lg_R_err;     // uncertainty in log10(R)
    vector[N] feh_obs;               // [Fe/H] offset from reference star (0 if not observed)
    array[N] int<lower=0, upper=1> has_feh_obs;  // 1 if star has Fe/H observation
}
parameters {
    real fac;
    real x;
    real y;
    real z;
    real z2;
    vector[N] lg_teff_true;
    vector[N] lg_R_true;
    vector[N] lg_amp_true;
    vector[N] lg_charb_true;
    vector<lower=-4, upper=1>[N] feh_true;
    real<lower=0> sigma;
}
model {
    vector[N] lg_R_pred;
    vector[N] feh_effective;

    for (i in 1:N) {
        if (has_feh_obs[i] == 1) {
            feh_effective[i] = feh_obs[i];
        } else {
            feh_effective[i] = feh_true[i];
            feh_true[i] ~ normal(-0.10, 0.4);
        }
    }

    lg_R_pred = fac + x * lg_amp_true + y * lg_charb_true + z * lg_teff_true + z2 * feh_effective;
    lg_R_pred ~ normal(lg_R_true, sigma);

    lg_R_obs ~ normal(lg_R_true, lg_R_err);
    lg_teff_obs ~ normal(lg_teff_true, lg_teff_err);

    for (i in 1:N) {
        vector[2] obs;
        vector[2] true_vals;
        matrix[2, 2] L;
        obs[1] = lg_amp_obs[i];
        obs[2] = lg_charb_obs[i];
        true_vals[1] = lg_amp_true[i];
        true_vals[2] = lg_charb_true[i];
        L = cholesky_decompose(gran_cov[i]);
        obs ~ multi_normal_cholesky(true_vals, L);
    }

    fac ~ normal(0, 2);
    x ~ normal(0, 0.5);
    y ~ normal(0, 0.5);
    z ~ normal(0, 0.5);
    z2 ~ normal(0, 0.5);
    sigma ~ normal(0, 0.2);
    lg_teff_true ~ normal(lg_teff_obs, 0.5);
    lg_R_true ~ normal(lg_R_obs, 0.5);
    lg_amp_true ~ normal(lg_amp_obs, 0.5);
    lg_charb_true ~ normal(lg_charb_obs, 0.5);
}
