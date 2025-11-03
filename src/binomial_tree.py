import numpy as np

class BinomialTreeSimulator:

    def set_binomial_tree_CRR(self):
        # set self.u, self.d, self.p according to Cox-Ross-Rubinstein model
        self.u = np.exp(self.sigma * np.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability

    def set_binomial_tree_JR(self):
        # set self.u, self.d, self.p according to Jarrow-Rudd model
        self.u = np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt))  # Up factor
        self.d = np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt - self.sigma * np.sqrt(self.dt))  # Down factor
        self.p = 0.5

    def set_binomial_tree_JR_risk_neutral(self):
        # set self.u, self.d, self.p according to Jarrow-Rudd risk-neutral model
        # Unlike the standard JR model, this one adjusts for risk-neutral drift
        self.u = np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt))  # Up factor
        self.d = np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt - self.sigma * np.sqrt(self.dt))  # Down factor
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability

    def set_binomial_tree_Tian(self):
        # set self.u, self.d, self.p according to Tian model
        nu = np.exp(self.sigma**2 * self.dt)
        self.u = 0.5 * np.exp(self.r * self.dt) * nu * (nu + 1 + np.sqrt(nu**2 + 2 * nu - 3))
        self.d = 0.5 * np.exp(self.r * self.dt) * nu * (nu + 1 - np.sqrt(nu**2 + 2 * nu - 3))
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability

    def set_binomial_tree(self, model='CRR'):
        if model == 'CRR':
            self.set_binomial_tree_CRR()
        elif model == 'JR':
            self.set_binomial_tree_JR()
        elif model == 'JR_risk_neutral':
            self.set_binomial_tree_JR_risk_neutral()
        elif model == 'Tian':
            self.set_binomial_tree_Tian()
        else:
            raise ValueError("Currently only 'CRR' model is implemented.")

    def __init__(self, S0, K, T, r, sigma, n_steps, option_type='put', model ='CRR'):
        self.S0 = S0  # Initial stock price
        self.K = K    # Strike price
        self.T = T    # Time to maturity
        self.r = r    # Risk-free rate
        self.sigma = sigma  # Volatility
        self.n_steps = n_steps  # Number of steps in the binomial tree
        self.option_type = option_type.lower()  # 'put' or 'call'
        self.dt = T / n_steps  # Time increment
        self.set_binomial_tree(model)  # Set up the binomial tree parameters
        self.asset_prices = self.create_binomial_tree() # Create the binomial tree
        self.option_prices = self.create_european_option_prices() # Assign option prices for each node in binomial tree

    def create_binomial_tree(self):
        # This creates an array storing asset prices at each node
        # This array is triangular in shape—at each timestep i, there are i+1 nodes
        # Here, we use a 2D array with padding for simplicity
        asset_prices = np.zeros((self.n_steps + 1, self.n_steps + 1))
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                asset_prices[i, j] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
        return asset_prices
    
    def calculate_payoff(self, stock_price):
        """Calculate option payoff at expiration"""
        if self.option_type == 'put':
            return max(self.K - stock_price, 0)
        elif self.option_type == 'call':
            return max(stock_price - self.K, 0)
        else:
            raise ValueError("option_type must be 'put' or 'call'")

    def create_european_option_prices(self):
        # This computes the European option price at each node using backward induction
        option_prices = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # Compute option values at maturity
        for j in range(self.n_steps + 1):
            option_prices[self.n_steps, j] = self.calculate_payoff(self.asset_prices[self.n_steps, j])
        
        # Backward induction to calculate option prices at earlier nodes
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                expected_value = (self.p * option_prices[i + 1, j] + (1 - self.p) * option_prices[i + 1, j + 1]) * np.exp(-self.r * self.dt)
                option_prices[i, j] = expected_value

        return option_prices
    
    def price_european_option(self):    
        # Price a European option using the binomial tree
        european_option_price = self.option_prices[0, 0]
        return european_option_price
        
    def price_american_option(self):
        # Price the American option using the binomial tree with early exercise feature
        # We examine at each node whether the option should be exercised or held
        american_option_prices = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # First, set the terminal condition (values at maturity) - same as European
        for j in range(self.n_steps + 1):
            american_option_prices[self.n_steps, j] = self.calculate_payoff(self.asset_prices[self.n_steps, j])
        
        # Backward induction to calculate option prices at earlier nodes
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Calculate continuation value using the American option prices we're computing
                continuation_value = (self.p * american_option_prices[i + 1, j] + 
                                    (1 - self.p) * american_option_prices[i + 1, j + 1]) * np.exp(-self.r * self.dt)
                exercise_value = self.calculate_payoff(self.asset_prices[i, j])
                american_option_prices[i, j] = max(continuation_value, exercise_value)
        
        return american_option_prices[0, 0]
    
    def american_premium(self):
        return self.price_american_option() - self.price_european_option()