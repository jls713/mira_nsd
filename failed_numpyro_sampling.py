## FAILED attempt at sampling from models with numpyro -- I think the problem is that the action calculation is not differentiable
# import numpyro
# import jax.numpy as jnp
# from df_jax import binney_df_jax_spline


# def action_calc_host(x):
#   # call a numpy (not jax.numpy) operation:
#   return sGM.af(agama_GalactocentricFromGalactic(x)).astype(x.dtype)

# def action_calc(x):
#   result_shape = jax.ShapeDtypeStruct([x.shape[0],3], x.dtype)
#   return jax.pure_callback(action_calc_host, result_shape, x)

# def sampling_lnL(data, params, aux_params):

#     coords = jnp.vstack([data['l_rad'], 
#                         data['b_rad'],
#                         params['s_samples'],
#                         params['pml_samples'],
#                         params['pmb_samples'],
#                         params['vlos_samples']]).T
    
#     actions = action_calc(coords)

#     return 4.*np.log(params['s_samples']) + np.log(np.cos(data['b_rad'])) + np.log(aux_params['bdf'](actions, data['log10P']))

# def generate_samples(best_params, aux_params, Ncopies):
    
#     Ndata = len(data['l'])    
    
#     bdf = binney_df_jax_spline(best_params['ln_Rdisk_coeffs'], 
#                                best_params['ln_Hdisk_coeffs'], 
#                                best_params['ln_sigmar0_coeffs'],
#                                aux_knots=aux_params['aux_knots'], 
#                                mass=1., 
#                                vO=aux_params['vO'], 
#                                Jv0=aux_params['Jv0'], 
#                                Jd0=aux_params['Jd0'])
    
#     data_here = {'l_rad':jnp.array(np.repeat(data['l_rad'].values, Ncopies)),
#                   'b_rad':jnp.array(np.repeat(data['b_rad'].values, Ncopies)),
#                   'log10P':jnp.array(np.repeat(np.log10(data['period'].values), Ncopies))}
    
#     default_dist = numpyro.distributions.Normal(5.).expand([jnp.array(int(Ndata * Ncopies))])

#     parameters = {'s_samples': default_dist,
#                   'pml_samples': default_dist,
#                   'pmb_samples': default_dist,
#                   'vlos_samples': default_dist}

#     model = numpyro_model(sampling_lnL, parameters, data_here, {'bdf':bdf})
#     model.run_mcmc(num_warmup=500, num_samples=500, num_chains=1)
#     return model
    