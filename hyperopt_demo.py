from hyperopt import fmin, tpe, hp

def f(x):
    return x**2 + x + 1

space = hp.uniform('x', -2, 2)

best = fmin(
    fn=f,
    space=space,
    algo=tpe.suggest,
    max_evals=1000
)

print(f"Optimal value of x: {best}")
