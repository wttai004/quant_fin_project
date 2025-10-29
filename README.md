# American option

American options are options where the owner can choose to exercise them at _any_ point up to and including the expiration date. Because this provides an additional mechanism to exercise compared to European options (which only allow exercise at the expiration date), American options are at least as valuable as their European counterparts, and their difference depends on the value of early exercise, called the _early exercise premium_. 

The premium depends on various factors. In the Black-Scholes model, American and European call options for non-dividend-paying stocks are generally priced the same (for dividend-paying stocks, deep-in-the-money option holders may wish to exercise before a dividend date to obtain dividends). However, put options can have substantial differences when it makes sense for the option holder to exercise early and invest the income at prevailing interest rates.

This project aims to systematically study American options via several calculations in the corresponding notebooks below:

## 1: perpetual_american_option

This notebook analyzes a perpetual American option, where the option has an infinite contract length and admits an analytical solution to its value. Here, I performed a Monte-Carlo simulation and compared its value against the theoretical value of the option at a no-arbitarge limit. I showed that the optimal strategy for American put option involves early exercise, while an American call option would not be exercised early. 

## 2: finite_american_option

For a finite American option, there is no analytical solution of the option value. Here, I empoloyed two methods: Cox-Ross-Rubinstein (CRR) binomial tree and Longstaff-Schwartz algorithm, both techniques used in industry to numerically simulate the value of an American put option. 

## 3: real_life_option_analysis

This notebook analyses real-life option data on several American non-dividend-paying securities and evaluates their implied volatility using Black-Scholes method for call options and CRR binomial tree for put options, otaining a volatility smile pattern. It also estimates the American premium of the put options, showing an increase in the premium for longer contracts. 

