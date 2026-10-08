# """This is a file to help design a sun terminator orbit for asteroid bennu.
# The eventual goal is to make a communication aware swarm 
# """

# import numpy as np
# from scipy.linalg import expm


# # "The OSIRIS-REx spacecraft is inserted into an orbit about Bennu (called Orbit B) approximately 3.5 years after launch, 
# # and remains in this orbit for about 60 days. The primary activity of OSIRIS-REx
# # during Orbit B is to gather optical data for 12 potential touch-and-go (TAG) sample collection
# # sites at 5 cm resolution, and to perform high fidelity mapping of Bennu’s gravity field. The
# # nominal Orbit B trajectory is a near-circular 1~km orbit about Bennu
# #, and is oriented about Bennu such that the orbit normal is parallel to the
# # Bennu-Sun direction (i.e. a terminator orbit). "

# # [1] ORBIT STABILITY OF OSIRIS-REX IN THE VICINITY OF BENNU USING A HIGH-FIDELITY SOLAR RADIATION MODEL
# # [2] The OSIRIS-REx target asteroid (101955) Bennu: Constraints on its physical, geological, and dynamical nature from astronomical observations

# # right_ascension_of_ascending_node = \pm 90
# # inclination = 90
# # eccentricity = 0
# # the terminator orbit should cross the plane twice per period

# AU = 149597870691 # m

# MU_SUN = 1.32712428e20 # m^3/s^2 Vallado



# sc_mass = 1198 #kg [1]
# sc_area = 12 #m^2 [1]
# sc_cr = 1.4 # [1]
# bennu_mu = 4.89044967462 #m^3
# bennu_j2 = 0.019261012209376163 # Chesley
# bennu_c22 = 0.00306499464152612 # Chesley

# semi_major = 1000
# bennu_aphelion = 1.355887692  #AU [2]
# bennu_perihelion = 0.896894360 # AU [2]

# a = 0.5 * (bennu_perihelion + bennu_aphelion) * AU
# e = (bennu_aphelion - bennu_perihelion) / (bennu_perihelion + bennu_aphelion)
# n_helio = np.sqrt(MU_SUN / a**3)

# def solve_Keplers_equation(M, e, tol=1e-12):
#     E = M if e < 0.8 else np.pi
#     for i in range(50):
#         dE = (E - e* np.sin(E) - M) / (1 - e*np.cos(E))
#         E -= dE
#         if abs(dE) < tol:
#             break
#     return E

# def sun_state(t, t_peri=0.0):
#     M = np.mod(n_helio * (t - t_peri), 2*np.pi)
#     E = solve_Keplers_equation(M,e)
#     r = a * (1 - e*np.cos(E))
#     nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
#     return r, nu

# def srp_accel(t, Cr, A, m, t_peri):
#     r, _ = sun_state(t, t_peri)
#     P = (1361.0 / 299792458) * (AU / r)**2
#     return Cr * P * A / m

# def sun_pos_from_bennu(t, t_peri):
#     r, nu = sun_state(t, t_peri)
#     return -r * np.array

# # frozen orbit is necessary here

# # drop the semimajor axis to less than 2.5 km





# def tan_lambda(mu_sun, mu_asteroid, asteroid_semi_major_axis, asteroid_eccentricity, sc_reflectance, sc_area_mass_ratio, sc_semi_major_axis):
#     P0 = 1361.0 / 299792458 * 149597870691
#     left = (3 * (1 + sc_reflectance) * P0) / (2 * sc_area_mass_ratio)
#     right = np.sqrt(sc_semi_major_axis / (mu_asteroid * mu_sun * asteroid_semi_major_axis * (1 - asteroid_eccentricity**2)))
#     return left * right

# def escape_limit(mu_asteroid, alpha):
#     return np.sqrt(3) / 4 * np.sqrt(mu_asteroid / alpha)



# def skew(v):
#     return np.array([
#         [0, -v[2], v[1]],
#         [v[2], 0, -v[0]],
#         [-v[1], v[0], 0]
#     ])


# def secular_matrix(tanL, Z_HAT, D_HAT):
#     Z, D = skew(Z_HAT), skew(D_HAT)
#     return np.array([
#         [-Z, tanL * D],
#         [tanL * D, -Z]
#     ])

# def secular_propagate(e0, h0, dnu, tanL, Z_HAT, D_HAT):
#     x = expm(secular_matrix(tanL, Z_HAT, D_HAT) * dnu) @ np.concatenate([e0, h0])
#     return x[:3], x[3:]

# def h_from_elements(i, Om):   # Eq. 7 with e=0
#     return np.array([np.sin(Om)*np.sin(i), -np.cos(Om)*np.sin(i), np.cos(i)])


# def circular_initial_state(node_deg):
#     O = np.radians(node_deg)
#     return np.zeros(3), np.array([np.sin(O), -np.cos(O), 0.0])



# def terminator_design(a_m):
#     L = np.arctan(tan_lambda)
#     e_max = np.sin(2 * L)