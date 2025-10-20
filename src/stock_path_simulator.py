import numpy as np

def _GBM_paths(S0, sigma, t, r, mu, n_sims, n_steps):
    """
    Simulates stock paths as geometric Brownian motions.
    
    Parameters:
    -----------
    S0 : float
        Underlying stock price at time 0
    sigma : float
        Yearly volatility
    t : float
        Time to expiration (years)
    r : float
        Risk-free interest rate
    mu : float
        Drift of log-returns
    n_sims : int
        Number of simulated paths
    n_steps : int
        Number of steps in each simulated path
    
    Returns:
    --------
    np.array
        Array of stock paths with shape (n_sims, n_steps+1)
    """
    dt = t / n_steps
    noise = np.random.normal(loc=0, scale=1, size=(n_sims, n_steps))
    log_returns = (mu + r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * noise
    exponent = np.cumsum(log_returns, axis=1)
    paths = S0 * np.exp(exponent)
    paths_with_start = np.insert(paths, 0, S0, axis=1)
    
    return paths_with_start



def _put_option_payoff(S, K, r, t):
    """
    Calculate the discounted payoff of a put option.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    t : float
        Time to expiration
    
    Returns:
    --------
    float
        Discounted put option payoff
    """
    return max(K - S, 0) * np.exp(-r * t)


def _call_option_payoff(S, K, r, t):
    """
    Calculate the discounted payoff of a call option.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    t : float
        Time to expiration
    
    Returns:
    --------
    float
        Discounted call option payoff
    """
    return max(S - K, 0) * np.exp(-r * t)

class StockPathSimulator:
    def __init__(self, S0=140, sigma=0.3, T=10, r=0.035, mu=0.0, n_sims=1000, n_steps=200):
        """
        Initialize the Stock Path Simulator.
        
        Parameters:
        -----------
        S0 : float, default=140
            Initial stock price
        sigma : float, default=0.3
            Volatility
        T : float, default=10
            Time horizon (years)
        r : float, default=0.035
            Risk-free rate
        mu : float, default=0.0
            Drift parameter
        n_sims : int, default=1000
            Number of simulations
        n_steps : int, default=200
            Number of time steps
        """
        self.S0 = S0
        self.sigma = sigma
        self.T = T
        self.r = r
        self.mu = mu
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.stock_paths = self.simulate_paths()

    def put_option_payoff(self, S, K, T):
        return _put_option_payoff(S, K, self.r, T)

    def call_option_payoff(self, S, K, T):
        return _call_option_payoff(S, K, self.r, T)

    def simulate_paths(self):
        self.stock_paths = _GBM_paths(self.S0, self.sigma, self.T, self.r, self.mu, self.n_sims, self.n_steps)
        return self.stock_paths