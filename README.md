# American option






## 1_perpetual_american_option: 

This notebook analyzes a perpetual American option, where the option has an infinite contract length and admits an analytical solution to its value. Here, I performed a Monte-Carlo simulation and compared its value against the theoretical value of the option at a no-arbitarge limit. I showed that the optimal strategy for American put option involves early exercise, while an American call option would not be exercised early. 

## 2_finite_american_option:

For a finite American option, there is no analytical solution of the option value. Here, I empoloyed two methods: Cox-Ross-Rubinstein (CRR) binomial tree and Longstaff-Schwartz algorithm, both techniques used in industry to numerically simulate the value of an American put option. 

## 3_real_life_option_analysis

This notebook analyses real-life option data on several American non-dividend-paying securities and evaluates their implied volatility using Black-Scholes method for call options and CRR binomial tree for put options, otaining a volatility smile pattern. It also estimates the American premium of the put options, showing an increase in the premium for longer contracts. 

