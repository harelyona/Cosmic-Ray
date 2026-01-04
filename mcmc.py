import numpy as np
import matplotlib.pyplot as plt

# --- 1. יצירת הדאטה (זהה למקודם) ---
np.random.seed(42)
true_A, true_omega, true_phi = 3.0, 2.0, 0.5
t = np.linspace(0, 10, 100)


def model(t, A, omega, phi):
    return A * np.sin(omega * t + phi)


y_true = model(t, true_A, true_omega, true_phi)
noise_sigma = 0.5
y_obs = y_true + np.random.normal(0, noise_sigma, size=len(t))


# --- 2. פונקציות Likelihood (זהה למקודם) ---
def log_likelihood(theta, t, y):
    A, omega, phi = theta
    y_model = model(t, A, omega, phi)
    sse = np.sum((y - y_model) ** 2)
    return -0.5 * sse / (noise_sigma ** 2)


def log_prior(theta):
    A, omega, phi = theta
    if 0 < A < 10 and 0 < omega < 5 and -np.pi < phi < np.pi:
        return 0.0
    return -np.inf


def log_posterior(theta, t, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y)


# --- 3. אלגוריתם MCMC עם Simulated Annealing ---
def run_mcmc_annealing(start_theta, iterations, step_size, t, y):
    chain = np.zeros((iterations, 3))
    chain[0] = start_theta
    current_log_prob = log_posterior(start_theta, t, y)

    # === פרמטרים של Annealing ===
    initial_temp = 100.0  # מתחילים "חם" מאוד
    cooling_rate = 0.9995  # קצב קירור (צריך להיות איטי)
    min_temp = 1.0  # הטמפרטורה הסופית (MCMC רגיל)
    current_temp = initial_temp

    accepted = 0

    for i in range(1, iterations):
        # הצעה לצעד חדש
        current_theta = chain[i - 1]
        proposal = current_theta + np.random.normal(0, step_size, 3)

        proposal_log_prob = log_posterior(proposal, t, y)

        # חישוב היחס
        diff = proposal_log_prob - current_log_prob

        # === התיקון של Annealing ===
        # מחלקים את ההפרש בטמפרטורה הנוכחית
        # אם הטמפרטורה גבוהה, ההבדל "מתגמד" וקל יותר לקבל צעדים גרועים
        adjusted_diff = diff / current_temp

        if np.log(np.random.rand()) < adjusted_diff:
            chain[i] = proposal
            current_log_prob = proposal_log_prob
            accepted += 1
        else:
            chain[i] = current_theta

        # === קירור ===
        # בכל צעד אנחנו מקררים מעט את המערכת
        current_temp = max(current_temp * cooling_rate, min_temp)

    print(f"Annealing Finished. Final Temp: {current_temp:.2f}")
    return chain


# --- 4. הרצה ---

# ניחוש התחלתי "גרוע" בכוונה!
# MCMC רגיל היה נכשל כאן, Annealing אמור להצליח
bad_start_guess = [1.0, 1.0, 0.0]

# הרצה עם Annealing
# הערה: נדרשות יותר איטרציות כדי לאפשר קירור איטי
iterations = 20000
step_size = 0.1

chain = run_mcmc_annealing(bad_start_guess, iterations, step_size, t, y_obs)

# חיתוך ה-Burn-in (לוקחים רק את הסוף כשהטמפרטורה ירדה)
burn_in = int(iterations * 0.5)
clean_chain = chain[burn_in:]

res_A = np.mean(clean_chain[:, 0])
res_omega = np.mean(clean_chain[:, 1])
res_phi = np.mean(clean_chain[:, 2])

print(f"True Params: A={true_A}, omega={true_omega}, phi={true_phi}")
print(f"Results:     A={res_A:.3f}, omega={res_omega:.3f}, phi={res_phi:.3f}")

# --- 5. ויזואליזציה ---
plt.figure(figsize=(10, 6))
plt.scatter(t, y_obs, label='Data', color='gray', alpha=0.5)
plt.plot(t, y_true, 'k--', label='True')
plt.plot(t, model(t, res_A, res_omega, res_phi), 'r-', label='Annealing Fit', lw=2)
plt.title(f'Simulated Annealing Result (Started from omega={bad_start_guess[1]})')
plt.legend()
plt.show()