"""
Black-Scholes Option Pricing Model
Implements closed-form solution for European options.
"""

import numpy as np
from scipy.stats import norm
from typing import Union


class BlackScholesModel:
    """
    Black-Scholes-Merton option pricing model for European options.
    
    The model assumes:
    - Log-normal distribution of stock prices
    - Constant volatility and interest rate
    - No dividends (can be extended)
    - Frictionless markets
    """
    
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        q: float = 0.0
    ):
        """
        Initialize Black-Scholes model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free interest rate (annualized)
            sigma: Volatility (annualized standard deviation)
            option_type: 'call' or 'put'
            q: Dividend yield (continuous, annualized)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.q = q
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input parameters."""
        if self.S <= 0:
            raise ValueError("Stock price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T < 0:
            raise ValueError("Time to maturity cannot be negative")
        if self.sigma < 0:
            raise ValueError("Volatility cannot be negative")
        if self.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
            
    def _d1(self) -> float:
        """Calculate d1 parameter in Black-Scholes formula."""
        if self.T == 0:
            return np.inf if self.S > self.K else -np.inf
        
        return (np.log(self.S / self.K) + 
                (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / \
               (self.sigma * np.sqrt(self.T))
    
    def _d2(self) -> float:
        """Calculate d2 parameter in Black-Scholes formula."""
        if self.T == 0:
            return np.inf if self.S > self.K else -np.inf
        
        return self._d1() - self.sigma * np.sqrt(self.T)
    
    def price(self) -> float:
        """
        Calculate option price using Black-Scholes formula.
        
        Returns:
            Option price
        """
        # Handle special case: at expiration
        if self.T == 0:
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)
        
        d1 = self._d1()
        d2 = self._d2()
        
        if self.option_type == 'call':
            price = (self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - 
                    self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:  # put
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - 
                    self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
        
        return price
    
    def delta(self) -> float:
        """
        Calculate Delta: ∂V/∂S
        Rate of change of option price with respect to underlying price.
        
        Returns:
            Delta value
        """
        if self.T == 0:
            if self.option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0
        
        d1 = self._d1()
        
        if self.option_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            return -np.exp(-self.q * self.T) * norm.cdf(-d1)
    
    def gamma(self) -> float:
        """
        Calculate Gamma: ∂²V/∂S²
        Rate of change of delta with respect to underlying price.
        
        Returns:
            Gamma value
        """
        if self.T == 0:
            return 0.0
        
        d1 = self._d1()
        return (np.exp(-self.q * self.T) * norm.pdf(d1)) / \
               (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        """
        Calculate Vega: ∂V/∂σ
        Sensitivity to volatility (per 1% change).
        
        Returns:
            Vega value (divided by 100 for 1% change)
        """
        if self.T == 0:
            return 0.0
        
        d1 = self._d1()
        return (self.S * np.exp(-self.q * self.T) * 
                norm.pdf(d1) * np.sqrt(self.T)) / 100
    
    def theta(self) -> float:
        """
        Calculate Theta: ∂V/∂t
        Time decay (per day, negative for long positions).
        
        Returns:
            Theta value (per day)
        """
        if self.T == 0:
            return 0.0
        
        d1 = self._d1()
        d2 = self._d2()
        
        term1 = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * 
                  self.sigma) / (2 * np.sqrt(self.T))
        
        if self.option_type == 'call':
            term2 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
            term3 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            term2 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
            term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        # Return per day (divide by 365)
        return (term1 + term2 + term3) / 365
    
    def rho(self) -> float:
        """
        Calculate Rho: ∂V/∂r
        Sensitivity to interest rate (per 1% change).
        
        Returns:
            Rho value (divided by 100 for 1% change)
        """
        if self.T == 0:
            return 0.0
        
        d2 = self._d2()
        
        if self.option_type == 'call':
            return (self.K * self.T * np.exp(-self.r * self.T) * 
                    norm.cdf(d2)) / 100
        else:
            return (-self.K * self.T * np.exp(-self.r * self.T) * 
                    norm.cdf(-d2)) / 100
    
    def all_greeks(self) -> dict:
        """
        Calculate all Greeks at once.
        
        Returns:
            Dictionary with all Greeks
        """
        return {
            'price': self.price(),
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }
    
    def intrinsic_value(self) -> float:
        """Calculate intrinsic value of the option."""
        if self.option_type == 'call':
            return max(0, self.S - self.K)
        else:
            return max(0, self.K - self.S)
    
    def time_value(self) -> float:
        """Calculate time value of the option."""
        return self.price() - self.intrinsic_value()
    
    def parity_check(self) -> dict:
        """
        Verify put-call parity: C - P = S - K*e^(-rT)
        
        Returns:
            Dictionary with call, put prices and parity values
        """
        call_price = BlackScholesModel(
            self.S, self.K, self.T, self.r, self.sigma, 'call', self.q
        ).price()
        
        put_price = BlackScholesModel(
            self.S, self.K, self.T, self.r, self.sigma, 'put', self.q
        ).price()
        
        left_side = call_price - put_price
        right_side = (self.S * np.exp(-self.q * self.T) - 
                     self.K * np.exp(-self.r * self.T))
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'left_side': left_side,
            'right_side': right_side,
            'difference': abs(left_side - right_side),
            'holds': abs(left_side - right_side) < 1e-6
        }
    
    def __repr__(self) -> str:
        """String representation of the option."""
        return (f"BlackScholes({self.option_type.upper()}, "
                f"S={self.S}, K={self.K}, T={self.T:.2f}, "
                f"r={self.r:.2%}, σ={self.sigma:.2%})")


def vectorized_price(
    S: np.ndarray,
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> np.ndarray:
    """
    Vectorized Black-Scholes pricing for multiple options.
    Efficient for computing price surfaces.
    
    Args:
        S: Stock price(s)
        K: Strike price(s)
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
    
    Returns:
        Array of option prices
    """
    # Ensure arrays
    S = np.atleast_1d(S)
    K = np.atleast_1d(K)
    T = np.atleast_1d(T)
    
    # Handle zero time
    mask = T > 0
    prices = np.zeros_like(S, dtype=float)
    
    if option_type == 'call':
        prices[~mask] = np.maximum(0, S[~mask] - K[~mask])
    else:
        prices[~mask] = np.maximum(0, K[~mask] - S[~mask])
    
    # Calculate for non-zero times
    if np.any(mask):
        d1 = (np.log(S[mask] / K[mask]) + 
              (r + 0.5 * sigma ** 2) * T[mask]) / (sigma * np.sqrt(T[mask]))
        d2 = d1 - sigma * np.sqrt(T[mask])
        
        if option_type == 'call':
            prices[mask] = (S[mask] * norm.cdf(d1) - 
                           K[mask] * np.exp(-r * T[mask]) * norm.cdf(d2))
        else:
            prices[mask] = (K[mask] * np.exp(-r * T[mask]) * norm.cdf(-d2) - 
                           S[mask] * norm.cdf(-d1))
    
    return prices
