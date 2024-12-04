import jax
import jax.numpy as jnp
from alpha2.expression.structure import Value


@jax.jit
def compute_metric(alpha: jnp.ndarray):
    """
    Compute the evaluation metric for the given alpha values.

    Parameters:
    alpha (jnp.ndarray): The alpha values to evaluate.

    Returns:
    float: The computed evaluation metric.
    """
    # Generate random returns for demonstration
    returns = jnp.array(
        [jnp.random.randn(alpha.shape[1]) for _ in range(alpha.shape[0])]
    )

    num_days = len(returns)
    return compute_ic(alpha, returns, num_days)


@jax.jit
def compute_ic(alpha: jnp.ndarray, returns: jnp.ndarray, num_days: int) -> float:
    """
    Compute the Information Coefficient (IC) between alpha values and returns.

    Parameters:
    alpha (jnp.ndarray): The alpha values.
    returns (jnp.ndarray): The market returns.
    num_days (int): Number of days.

    Returns:
    float: The average Information Coefficient.
    """
    ic = 0.0
    for d in range(num_days):
        cov = jnp.cov(alpha[d], returns[d])[0, 1]
        std_alpha = jnp.std(alpha[d])
        std_returns = jnp.std(returns[d])
        ic += cov / (std_alpha * std_returns)
    return ic / num_days


def fast_evaluate(alpha: Value):
    """
    Wrapper function to evaluate an alpha value.

    Parameters:
    alpha (Value): The alpha value to evaluate.

    Returns:
    float: The computed metric.
    """
    return compute_metric(alpha.value)
