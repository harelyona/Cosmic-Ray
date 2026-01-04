import numpy as np
import matplotlib.pyplot as plt

# --- קבועים ---
ITERATIONS = 40000       # הגדלתי מעט ליתר ביטחון
STEP_SIZE = 0.5          # <--- השינוי הקריטי: צעד גדול יותר לסריקה מהירה
BURN_IN_COEFF = 0.6      # זורקים את ה-60% הראשונים (שהיו חמים מדי)
INITIAL_TEMP = 5000.0
COOLING_RATE = 0.9997
MIN_TEMP = 0.5

# --- 1. יצירת הדאטה ---
np.random.seed(42)
true_A, true_omega, true_phi = 5, 8, 0.9
t = np.linspace(0, 10, 100)

def model(t, A, omega, phi):
    return A * np.sin(omega * t + phi)

y_true = model(t, true_A, true_omega, true_phi)
noise_sigma = 0.5
y_obs = y_true + np.random.normal(0, noise_sigma, size=len(t))

# --- 2. Likelihood & Prior ---
def log_likelihood(theta, t, y):
    A, omega, phi = theta
    y_model = model(t, A, omega, phi)
    sse = np.sum((y - y_model) ** 2)
    return -0.5 * sse / (noise_sigma ** 2)

def log_prior(theta):
    A, omega, phi = theta
    # וידוא שהתשובה (8) נמצאת בתוך הטווח
    if 0 < A < 10 and 0 < omega < 12 and -np.pi < phi < np.pi:
        return 0.0
    return -np.inf

def log_posterior(theta, t, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y)

# --- 3. MCMC Annealing ---
def run_mcmc_annealing(start_theta, iterations, step_size, t, y):
    chain = np.zeros((iterations, 3))
    chain[0] = start_theta
    current_log_prob = log_posterior(start_theta, t, y)

    current_temp = INITIAL_TEMP
    accepted = 0

    for i in range(1, iterations):
        current_theta = chain[i - 1]
        proposal = current_theta + np.random.normal(0, step_size, 3)

        proposal_log_prob = log_posterior(proposal, t, y)
        diff = proposal_log_prob - current_log_prob

        # Annealing Logic
        adjusted_diff = diff / current_temp

        if np.log(np.random.rand()) < adjusted_diff:
            chain[i] = proposal
            current_log_prob = proposal_log_prob
            accepted += 1
        else:
            chain[i] = current_theta

        current_temp = max(current_temp * COOLING_RATE, MIN_TEMP)

    print(f"Annealing Finished.")
    print(f"Final Temp: {current_temp:.2f}")
    print(f"Acceptance Rate: {accepted/iterations:.2%}") # אינדיקציה חשובה!
    return chain

# --- 4. הרצה ---
bad_start_guess = [1.0, 1.0, 0.0]

chain = run_mcmc_annealing(bad_start_guess, ITERATIONS, STEP_SIZE, t, y_obs)

burn_in = int(ITERATIONS * BURN_IN_COEFF)
clean_chain = chain[burn_in:]

res_A = np.mean(clean_chain[:, 0])
res_omega = np.mean(clean_chain[:, 1])
res_phi = np.mean(clean_chain[:, 2])

print("-" * 30)
print(f"True Params: A={true_A}, omega={true_omega}, phi={true_phi}")
print(f"Results:     A={res_A:.3f}, omega={res_omega:.3f}, phi={res_phi:.3f}")

# --- 5. ויזואליזציה ---
plt.figure(figsize=(12, 5))

# גרף התאמה
plt.subplot(1, 2, 1)
plt.scatter(t, y_obs, label='Data', color='gray', alpha=0.5)
plt.plot(t, y_true, 'k--', label='True')
plt.plot(t, model(t, res_A, res_omega, res_phi), 'r-', label='Annealing Fit', lw=2)
plt.title(f'Result (Step Size={STEP_SIZE})')
plt.legend()

# גרף Trace של אומגה - לראות איך הוא מצא את הדרך
plt.subplot(1, 2, 2)
plt.plot(chain[:, 1], color='green', alpha=0.6)
plt.axhline(true_omega, color='black', linestyle='--')
plt.title('Omega Trace (Search Process)')
plt.xlabel('Iterations')
plt.ylabel('Omega')

plt.tight_layout()
plt.show()