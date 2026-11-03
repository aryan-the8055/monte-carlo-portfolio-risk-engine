# monte-carlo-portfolio-risk-engine
Monte Carlo portfolio risk engine comparing Gaussian, Student-t, block bootstrap, and GARCH(1,1) return models. Simulates 10K+ paths to compute VaR, CVaR, and drawdown probability across 1/5/10-year horizons, then validates each model's predicted VaR against real market history with a Kupiec exception test.
