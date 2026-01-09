from typing import List, Callable, Sequence

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde

worked_time = 11
# --- CONSTANTS ---
ITERATIONS = 40000
STEP_SIZE = 0.5
BURN_IN_COEFF = 0.6
INITIAL_TEMP = 5000.0
COOLING_RATE = 0.9997
MIN_TEMP = 0.5
POTASSIUM_40_HALF_TIME = 1

# --- 1. DEFINE SYMBOLS GLOBALLY ---
# We use specific names so they don't clash with data variables later
t_sym, A, B, p_short, p_long, phi1, phi2, alpha = sp.symbols('t A B p_short p_long phi1 phi2 alpha')


def _get_symbolic_expr():
    """
    Internal helper: Defines the math ONE time using symbols.
    """
    # Use t_sym (symbol), NOT t_values (array)
    angle1 = 2 * sp.pi * t_sym / p_short + phi1
    angle2 = 2 * sp.pi * t_sym / p_long + phi2

    term1 = A * (1 + sp.sin(angle1)) ** alpha
    term2 = B * (1 + sp.sin(angle2))

    raw_signal_t = term1 + term2

    # Calculate constant 'c' at t=0
    raw_signal_0 = raw_signal_t.subs(t_sym, 0)
    c = 1 - raw_signal_0

    return raw_signal_t + c


def get_crf_model():
    """Returns the Flux Model Function (NumPy ready)"""
    expr = _get_symbolic_expr()
    func = sp.lambdify(
        (t_sym, A, B, p_short, p_long, phi1, phi2, alpha),
        expr,
        modules='numpy'
    )
    return func


def get_derivative_model():
    """Returns the Derivative Function dPhi/dt (NumPy ready)"""
    expr = _get_symbolic_expr()
    deriv_expr = sp.diff(expr, t_sym)  # Differentiate with respect to symbol t_sym
    func = sp.lambdify(
        (t_sym, A, B, p_short, p_long, phi1, phi2, alpha),
        deriv_expr,
        modules='numpy'
    )
    return func


# --- 2. LOGIC FUNCTIONS ---

def calc_ratios(time_values: np.ndarray, crf_function, crf_function_parameters) -> np.ndarray:
    """
    Calculates the ratio N40 / N41 for varying exposure ages.
    Assumes time_values are Lookback Time (0 = Today, 100 = 100 Ma ago).
    """
    flux = crf_function(time_values, *crf_function_parameters)
    decay_weight = np.exp(-time_values/POTASSIUM_40_HALF_TIME)
    n_40 = integrate.cumulative_trapezoid(flux * decay_weight, time_values, initial=0)
    n_41 = integrate.cumulative_trapezoid(flux, time_values, initial=0)
    ratio = np.zeros_like(n_40)
    mask = n_41 > 0
    ratio[mask] = n_40[mask] / n_41[mask]

    return ratio


def calc_exposure_age(t_values:np.ndarray, crf_function: Callable, crf_function_parameters:Sequence[float]) -> np.ndarray:
    """
    Solves for t using fsolve.
    Equation: ratio * t = tau * (1 - exp(-t/tau))
    """
    ratios = calc_ratios(t_values, crf_function, crf_function_parameters)
    exposure_ages = np.zeros_like(t_values)
    for i, ratio in enumerate(ratios):
        equation_to_solve = lambda t: ratio * t - POTASSIUM_40_HALF_TIME * (1 - np.exp(-t / POTASSIUM_40_HALF_TIME))
        exposure_ages[i] = fsolve(equation_to_solve, POTASSIUM_40_HALF_TIME)[0]
    return exposure_ages


def create_and_plot_pdf(exposure_ages: np.ndarray):
    """
    Creates a Probability Density Function (PDF) from exposure ages using KDE
    and plots the result.
    """
    # 1. Remove NaNs or Infs if the solver failed for some points
    valid_ages = exposure_ages[np.isfinite(exposure_ages)]
    # 2. Create the Kernel Density Estimator (KDE)
    # This object represents the PDF function
    kde = gaussian_kde(valid_ages)

    # 3. Define the X-axis grid (the range of ages we want to look at)
    # We go slightly below min and above max to see the tails of the curve
    x_grid = np.linspace(min(valid_ages) - 10, max(valid_ages) + 10, 500)

    # 4. Calculate the PDF values (Y-axis)
    pdf_values = kde(x_grid)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))

    # A. Plot the Smooth PDF line
    plt.plot(x_grid, pdf_values, color='red', label='KDE (Smooth PDF)', linewidth=2)

    # B. Plot a Histogram behind it (for verification)
    # density=True ensures the histogram is normalized so it matches the PDF height
    plt.hist(valid_ages, bins=30, density=True, alpha=0.3, color='blue', label='Histogram')

    plt.title("Probability Density Function of Exposure Ages")
    plt.xlabel("Exposure Age (Ma)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return x_grid, pdf_values


def log_likelihood(model_func, theta, t, y):
    # Calculate model predictions
    y_model = model_func(t, *theta)
    # Simple Sum of Squared Errors
    sse = np.sum((y - y_model) ** 2)
    return -0.5 * sse


def log_prior(theta):
    A_val, B_val, p_s, p_l, ph1, ph2, al = theta
    # Basic bounds check
    if 0 < A_val < 20 and 0 < B_val < 20:
        return 0.0
    return -np.inf


def log_posterior(model_func, theta, t, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(model_func, theta, t, y)


def run_mcmc_annealing(model_func, start_theta, iterations, step_size, t, y):
    # Initialize chain
    n_params = len(start_theta)
    chain = np.zeros((iterations, n_params))
    chain[0] = start_theta

    current_log_prob = log_posterior(model_func, start_theta, t, y)
    current_temp = INITIAL_TEMP

    for i in range(1, iterations):
        # Propose new state
        proposal = chain[i - 1] + np.random.normal(0, step_size, n_params)

        # Calculate probabilities
        proposal_log_prob = log_posterior(model_func, proposal, t, y)
        diff = proposal_log_prob - current_log_prob

        # Annealing acceptance logic
        if diff > 0 or np.log(np.random.rand()) < (diff / current_temp):
            chain[i] = proposal
            current_log_prob = proposal_log_prob
        else:
            chain[i] = chain[i - 1]

        # Cool down
        current_temp = max(current_temp * COOLING_RATE, MIN_TEMP)

    burn_in = int(iterations * BURN_IN_COEFF)
    return np.mean(chain[burn_in:], axis=0)


# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Time Array
    t_values = np.linspace(0, 200, 50)
    initial_crf_parameters = [0, 0, 100, 100, 0, 0, 1]

    # 2. Generate the model function ONCE
    # This creates the actual function that accepts (t, A, B...)
    model_func = get_crf_model()

    # 3. Pass 'model_func' (the result), NOT 'get_crf_model' (the factory)
    # The error occurred here because you passed the wrong function object
    ea = calc_exposure_age(t_values, model_func, initial_crf_parameters)

    # 4. Plot
    create_and_plot_pdf(ea)

