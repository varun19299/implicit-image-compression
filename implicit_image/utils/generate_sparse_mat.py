from scipy import sparse, stats

density = 0.1
uniform_rvs = stats.randint(0, 255).rvs

# Random sparse matrix
# with int8
mat = sparse.random(128, 128, density=density, format="csc", data_rvs=uniform_rvs)

print(f"Sparse CSC rep {mat}")

print(f"Dense Rep {mat.todense()}")
