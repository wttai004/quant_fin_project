# American option

American options are options where the owner can choose to exercise them at _any_ point up to and including the expiration date. Because this provides an additional mechanism to exercise compared to European options (which only allow exercise at the expiration date), American options are at least as valuable as their European counterparts, and their difference depends on the value of early exercise, called the _early exercise premium_. 

The premium depends on various factors. In the Black-Scholes model, American and European call options for non-dividend-paying stocks are generally priced the same (for dividend-paying stocks, deep-in-the-money option holders may wish to exercise before a dividend date to obtain dividends). However, put options can have substantial differences when it makes sense for the option holder to exercise early and invest the income at prevailing interest rates.

This project systematically studies American options through analytical solutions, numerical methods, and real-world data analysis across three comprehensive notebooks:

## 1: Perpetual American Option

This notebook analyzes perpetual American options with infinite contract length, which admit closed-form analytical solutions. Key findings include:

- **Monte Carlo validation**: Validated optimal exercise boundaries for perpetual puts ($L_* = \frac{2r}{2r+\sigma^2}K$) and calls (infinite boundary) via  simulation, showing that put options should be exercised when the stock price falls to $L_*$
- **Computational optimization**: Benchmarked simulation parameters (time horizon T and discretization steps n_steps) to identify efficient accuracy-performance trade-offs

## 2: Finite American Option

For finite-maturity American options without analytical solutions, this notebook implements and compares multiple numerical methods:

- **Binomial tree models**: Implemented Cox-Ross-Rubinstein (CRR), Jarrow-Rudd (JR), and Tian models with convergence analysis
- **Monte Carlo simulation**: Developed Longstaff-Schwartz regression-based algorithm for benchmark pricing
- **Put-call parity**: Demonstrated how early exercise breaks classical put-call parity for American options
- **Early exercise premium**: Quantified the American premium across different strikes, volatilities, and maturities
- **Model comparison**: Conducted benchmarking, showing CRR remains the preferred choice due to simplicity and competitive accuracy, with Tian's higher-moment matching providing advantages only in specific regimes (low moneyness, short maturity, high volatility)
- **Practical recommendation**: Sufficient discretization steps (n_steps ≥ 250) matter more than model choice for accuracy

## 3: Real Life Option Analysis

This notebook analyzes real-world option data from OptionMetrics Ivy DB US for multiple non-dividend-paying securities (AMZN, MSFT, NVDA, TSLA, META):

- **Implied volatility analysis**: Calculated IV using Black-Scholes for call options and CRR binomial tree for put options
- **Volatility smile**: Documented the characteristic smile pattern across moneyness, with IV increasing for deep ITM and OTM options
- **Volatility term structure**: Examined how implied volatility varies with time to expiration for at-the-money options
- **Model validation**: Compared calculated IVs against dataset values, demonstrating strong agreement and validating our implementations
- **American vs European**: Analyzed relative errors between Black-Scholes and binomial tree IV estimates, showing close agreement for OTM/ATM puts but divergence for deep ITM puts where early exercise matters
- **American premium estimation**: Quantified the early exercise premium for puts, showing it increases with strike price and becomes most valuable for deep in-the-money options