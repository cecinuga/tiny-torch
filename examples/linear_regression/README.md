# Polynomial Regression

# Differences between polynomial:
The `in_feature` of the `Linear layer` must be equa to the rank of the polynomial, the model should be able to learn the ingredients that form a polynomial:

- Linear case: `Linear(in_feature=1, out_feature=1)`
- Quadratic case: `Linear(in_feature=2, out_feature=1)`
- Cubic case: `Linear(in_feature=3, out_feature=1)`

One `in_feature` for each of the degree of freedom of the function we want estimate.
