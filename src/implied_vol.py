"""
Implied Volatility Solver
Uses Newton-Raphson method for fast convergence.
"""

import numpy as np
from typing import Optional
from .black_scholes import BlackScholesModel


class ImpliedVolatilitySolver:
    """
    Solver for implied volatility using Newton-Raphson method.
    """
    
    def __init__(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        q: float = 0.0
    ):
        """
        Initialize IV solver.
        
        Args:
            market_price: Observed market price of option
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            q: Dividend yield
        """
        self.market_price = market_price
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.option_type = option_type
        self.q = q
        
    def solve(
        self,
        initial_guess: float = 0.2,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """
        Solve for implied volatility using Newton-Raphson.
        
        Args:
            initial_guess: Starting volatility estimate
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility or None if no convergence
        """
        sigma = initial_guess
        
        for i in range(max_iterations):
            # Calculate option price with current sigma
            option = BlackScholesModel(
                self.S, self.K, self.T, self.r, sigma,
                self.option_type, self.q
            )
            
            price = option.price()
            vega = option.vega() * 100  # Vega is per 1%, scale back
            
            # Check convergence
            price_diff = price - self.market_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            # Newton-Raphson update
            if vega < 1e-10:  # Avoid division by very small numbers
                return None
            
            sigma = sigma - price_diff / vega
            
            # Keep sigma positive and reasonable
            sigma = max(0.001, min(sigma, 5.0))
        
        # Did not converge
        return None
    
    def solve_bisection(
        self,
        lower_bound: float = 0.001,
        upper_bound: float = 5.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """
        Solve using bisection method (more robust, slower).
        
        Args:
            lower_bound: Lower bound for sigma
            upper_bound: Upper bound for sigma
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility or None if no solution
        """
        for i in range(max_iterations):
            mid = (lower_bound + upper_bound) / 2
            
            option = BlackScholesModel(
                self.S, self.K, self.T, self.r, mid,
                self.option_type, self.q
            )
            
            price = option.price()
            diff = price - self.market_price
            
            if abs(diff) < tolerance:
                return mid
            
            if diff > 0:
                upper_bound = mid
            else:
                lower_bound = mid
                
            if upper_bound - lower_bound < tolerance:
                return mid
        
        return None


def calculate_iv_surface(
    market_prices: np.ndarray,
    S: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    option_type: str = 'call'
) -> np.ndarray:
    """
    Calculate implied volatility surface.
    
    Args:
        market_prices: 2D array of market prices [strikes × maturities]
        S: Current stock price
        strikes: Array of strike prices
        maturities: Array of times to maturity
        r: Risk-free rate
        option_type: 'call' or 'put'
        
    Returns:
        2D array of implied volatilities
    """
    iv_surface = np.zeros_like(market_prices)
    
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            solver = ImpliedVolatilitySolver(
                market_prices[i, j],
                S, K, T, r, option_type
            )
            
            iv = solver.solve()
            iv_surface[i, j] = iv if iv is not None else np.nan
    
    return iv_surface


def volatility_smile(
    market_prices: np.ndarray,
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    option_type: str = 'call'
) -> np.ndarray:
    """
    Calculate volatility smile for a given maturity.
    
    Args:
        market_prices: Array of market prices for different strikes
        S: Current stock price
        strikes: Array of strike prices
        T: Time to maturity
        r: Risk-free rate
        option_type: 'call' or 'put'
        
    Returns:
        Array of implied volatilities
    """
    ivs = np.zeros_like(strikes)
    
    for i, (K, price) in enumerate(zip(strikes, market_prices)):
        solver = ImpliedVolatilitySolver(S, K, T, r, price, option_type)
        iv = solver.solve()
        ivs[i] = iv if iv is not None else np.nan
    
    return ivs
