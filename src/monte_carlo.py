"""
Monte Carlo Simulation for Option Pricing
Supports European and path-dependent options.
"""

import numpy as np
from typing import Callable, Optional


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for option pricing.
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        num_simulations: int = 100000,
        num_steps: int = 252,
        seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo engine.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            num_simulations: Number of simulation paths
            num_steps: Number of time steps per path
            seed: Random seed for reproducibility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        
        if seed is not None:
            np.random.seed(seed)
            
        self.dt = T / num_steps
        self.paths = None
        
    def generate_paths(self) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion.
        
        Returns:
            Array of shape (num_simulations, num_steps + 1)
        """
        # Generate random shocks
        Z = np.random.standard_normal((self.num_simulations, self.num_steps))
        
        # Initialize paths
        paths = np.zeros((self.num_simulations, self.num_steps + 1))
        paths[:, 0] = self.S0
        
        # Simulate paths using exact solution
        for t in range(1, self.num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * self.dt +
                self.sigma * np.sqrt(self.dt) * Z[:, t-1]
            )
        
        self.paths = paths
        return paths
    
    def price_european_call(self) -> tuple[float, float]:
        """
        Price European call option.
        
        Returns:
            (price, standard_error)
        """
        if self.paths is None:
            self.generate_paths()
        
        # Terminal payoffs
        payoffs = np.maximum(self.paths[:, -1] - self.K, 0)
        
        # Discount to present value
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        # Standard error
        std_error = np.std(payoffs) / np.sqrt(self.num_simulations)
        std_error *= np.exp(-self.r * self.T)
        
        return price, std_error
    
    def price_european_put(self) -> tuple[float, float]:
        """
        Price European put option.
        
        Returns:
            (price, standard_error)
        """
        if self.paths is None:
            self.generate_paths()
        
        # Terminal payoffs
        payoffs = np.maximum(self.K - self.paths[:, -1], 0)
        
        # Discount to present value
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        # Standard error
        std_error = np.std(payoffs) / np.sqrt(self.num_simulations)
        std_error *= np.exp(-self.r * self.T)
        
        return price, std_error
    
    def price_asian_call(self, average_type: str = 'arithmetic') -> tuple[float, float]:
        """
        Price Asian call option (average price option).
        
        Args:
            average_type: 'arithmetic' or 'geometric'
            
        Returns:
            (price, standard_error)
        """
        if self.paths is None:
            self.generate_paths()
        
        if average_type == 'arithmetic':
            avg_prices = np.mean(self.paths, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(self.paths), axis=1))
        
        payoffs = np.maximum(avg_prices - self.K, 0)
        
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.num_simulations)
        std_error *= np.exp(-self.r * self.T)
        
        return price, std_error
    
    def price_lookback_call(self) -> tuple[float, float]:
        """
        Price lookback call option (strike = minimum price).
        
        Returns:
            (price, standard_error)
        """
        if self.paths is None:
            self.generate_paths()
        
        min_prices = np.min(self.paths, axis=1)
        payoffs = self.paths[:, -1] - min_prices
        
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.num_simulations)
        std_error *= np.exp(-self.r * self.T)
        
        return price, std_error
    
    def price_barrier_call(
        self,
        barrier: float,
        barrier_type: str = 'up-and-out'
    ) -> tuple[float, float]:
        """
        Price barrier call option.
        
        Args:
            barrier: Barrier level
            barrier_type: 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
            
        Returns:
            (price, standard_error)
        """
        if self.paths is None:
            self.generate_paths()
        
        terminal_payoffs = np.maximum(self.paths[:, -1] - self.K, 0)
        
        if barrier_type == 'up-and-out':
            # Option knocked out if price goes above barrier
            knocked_out = np.any(self.paths >= barrier, axis=1)
            payoffs = np.where(knocked_out, 0, terminal_payoffs)
            
        elif barrier_type == 'down-and-out':
            # Option knocked out if price goes below barrier
            knocked_out = np.any(self.paths <= barrier, axis=1)
            payoffs = np.where(knocked_out, 0, terminal_payoffs)
            
        elif barrier_type == 'up-and-in':
            # Option activated if price goes above barrier
            knocked_in = np.any(self.paths >= barrier, axis=1)
            payoffs = np.where(knocked_in, terminal_payoffs, 0)
            
        else:  # down-and-in
            # Option activated if price goes below barrier
            knocked_in = np.any(self.paths <= barrier, axis=1)
            payoffs = np.where(knocked_in, terminal_payoffs, 0)
        
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.num_simulations)
        std_error *= np.exp(-self.r * self.T)
        
        return price, std_error
    
    def price_custom(self, payoff_function: Callable) -> tuple[float, float]:
        """
        Price option with custom payoff function.
        
        Args:
            payoff_function: Function that takes paths array and returns payoffs
            
        Returns:
            (price, standard_error)
        """
        if self.paths is None:
            self.generate_paths()
        
        payoffs = payoff_function(self.paths)
        
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.num_simulations)
        std_error *= np.exp(-self.r * self.T)
        
        return price, std_error
    
    def calculate_greeks_fd(
        self,
        option_type: str = 'call',
        epsilon: float = 0.01
    ) -> dict:
        """
        Calculate Greeks using finite differences.
        
        Args:
            option_type: 'call' or 'put'
            epsilon: Small change for finite difference
            
        Returns:
            Dictionary with Greeks
        """
        # Price at current parameters
        if option_type == 'call':
            V, _ = self.price_european_call()
        else:
            V, _ = self.price_european_put()
        
        # Delta: dV/dS
        self.S0 += epsilon
        self.paths = None
        if option_type == 'call':
            V_up, _ = self.price_european_call()
        else:
            V_up, _ = self.price_european_put()
        self.S0 -= epsilon
        delta = (V_up - V) / epsilon
        
        # Vega: dV/dsigma
        self.sigma += epsilon * 0.01  # 1% change
        self.paths = None
        if option_type == 'call':
            V_sigma, _ = self.price_european_call()
        else:
            V_sigma, _ = self.price_european_put()
        self.sigma -= epsilon * 0.01
        vega = (V_sigma - V) / (epsilon * 0.01)
        
        return {
            'price': V,
            'delta': delta,
            'vega': vega / 100  # Per 1% change
        }
