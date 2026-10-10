import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)     # random number generator with a fixed seed

def make_group(n, income, spending, age, visits):
    """Create n fake customers around the given average values."""
    return pd.DataFrame({
        "income_k": rng.normal(income, 8, n),
        "spending_score": np.clip(rng.normal(spending, 8, n), 1, 100),
        "age": np.clip(rng.normal(age, 8, n), 18, 80),
        "visits_per_month": np.clip(rng.normal(visits, 1.5, n), 0, None),
    })

# Four hidden customer types. We do NOT save which type each customer belongs to.
customers = pd.concat([
    make_group(150, income=25, spending=25, age=45, visits=3),
    make_group(150, income=90, spending=85, age=38, visits=12),
    make_group(150, income=45, spending=70, age=24, visits=9),
    make_group(150, income=60, spending=20, age=55, visits=2),
], ignore_index=True)

customers = customers.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle the rows
customers.to_csv("customers.csv", index=False)

print(customers.shape)
print(customers.describe().round(1))

plt.scatter(customers["income_k"], customers["spending_score"], alpha=0.6)
plt.xlabel("Income (thousand per year)")
plt.ylabel("Spending score")
plt.title("600 customers, no labels")
plt.show()