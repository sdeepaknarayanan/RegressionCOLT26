import sympy as sp

# Symbols
eps, s = sp.symbols('eps s', real=True)
k = sp.Rational(4, 5)

# Weights
w1 = 1 - eps
w2 = (1 - eps) / k**3
w3 = 1 - w1 - w2  # = eps - (1-eps)/k**3

# Means
mu1 = 1 / (3 * sp.sqrt(1 - eps))
mu2 = -k * mu1
mu3 = -(w1*mu1 + w2*mu2) / w3

# Moment pieces (matching your Mathematica)
sumMeansSq = w1*mu1**2 + w2*mu2**2 + w3*mu3**2
rhsVar  = 1 - sumMeansSq - w1*s
rhsSkew = -w3*mu3**3 - 3*w1*mu1*s

# Linear system for v2 and v3:
#   w2*v2 + w3*v3 = rhsVar
#   (w2*mu2)*v2 + (w3*mu3)*v3 = rhsSkew/3
v2, v3 = sp.symbols('v2 v3')

A = sp.Matrix([[w2,       w3],
               [w2*mu2,   w3*mu3]])
b = sp.Matrix([rhsVar, rhsSkew/3])

sol = sp.simplify(A.LUsolve(b))   # solves for v2, v3
v2_expr = sp.simplify(sol[0])
v3_expr = sp.simplify(sol[1])

print("v2(eps,s) =")
sp.pprint(v2_expr)
print("\nv3(eps,s) =")
sp.pprint(v3_expr)

# (Optional) verify the equations symbolically:
check1 = sp.simplify(w2*v2_expr + w3*v3_expr - rhsVar)
check2 = sp.simplify(w2*mu2*v2_expr + w3*mu3*v3_expr - rhsSkew/3)
print("\nchecks (should both be 0):", check1, check2)

# (Optional) domain-related helper:
epMin = 1 - k**3/(1 + k**3)  # matches your Mathematica epMin
print("\nepMin =", sp.nsimplify(epMin), "≈", float(epMin))
