import sympy as sp

# --- Step 1: Symbolic variables ---
eps, sig2, tau2 = sp.symbols('epsilon sigma^2 tau^2', real=True, positive=True)
mu_s, mu_2, sig1_sq = sp.symbols('mu_s mu_2 sigma_1^sq', real=True)

# Relations
mu_1 = -2 * mu_2
w_s, w_1, w_2 = 1 - eps, eps / 9, 8 * eps / 9

# --- Step 2: Moment equations ---
# Mean = 0
eq_E1 = sp.Eq(w_s*mu_s + w_1*mu_1 + w_2*mu_2, 0)

# Second moment = 1
eq_E2 = sp.Eq(w_s*(mu_s**2 + sig2) + w_1*(mu_1**2 + sig1_sq) + w_2*(mu_2**2 + tau2), 1)

# Third moment = 0
eq_E3 = sp.Eq(w_s*(mu_s**3 + 3*mu_s*sig2) +
              w_1*(mu_1**3 + 3*mu_1*sig1_sq) +
              w_2*(mu_2**3 + 3*mu_2*tau2), 0)

# --- Step 3: Eliminate variables ---
# From mean=0, solve for mu_s in terms of mu_2
mu_s_expr = sp.solve(eq_E1, mu_s)[0]

# Substitute into third moment, solve for sigma1^2
eq_E3_sub = eq_E3.subs(mu_s, mu_s_expr)
sig1_sq_expr = sp.solve(eq_E3_sub, sig1_sq)[0]

# Now substitute into second moment and solve for mu_2**2
eq_E2_sub = eq_E2.subs({mu_s: mu_s_expr, sig1_sq: sig1_sq_expr})
mu2_sq_expr = sp.solve(eq_E2_sub, mu_2**2)[0]

# --- Step 4: Print results ---
print("--- Symbolic Expressions ---")
print("mu_s (in terms of mu_2, eps, sig2):")
sp.pprint(mu_s_expr)

print("\nmu_1 (in terms of mu_2):")
sp.pprint(mu_1)

print("\nsigma_1^2 (in terms of mu_2, eps, sig2, tau2):")
sp.pprint(sig1_sq_expr.simplify())

print("\n|mu_2|^2 (closed form in eps, sig2, tau2):")
sp.pprint(mu2_sq_expr.simplify())

sigma_1_subs = sig1_sq_expr.subs({mu_2: sp.sqrt(mu2_sq_expr)})
sp.pprint(sigma_1_subs.simplify())
