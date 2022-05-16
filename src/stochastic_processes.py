import numpy as np
import logging
import sys


class StochasticProcess:
    def __init__(self, s, r, vol, t, carry, n):
        """
        :param float s: Spot price
        :param float r: Risk-free (domestic) rate
        :param float vol: Log-normal volatility
        :param float t: Time to expiration
        :param float carry: Cost of carry
        :param int n: Number of simulated paths
        """
        self.logger = logging.getLogger(__name__)
        self.s = s
        self.vol = vol
        self.t = t
        self.carry = carry
        self.n = n
        self.r = r

    def get_wiener_process(self):
        """Simulate Wiener Process with antithetic sampling"""
        wiener_process = np.random.randn(int(self.n / 2)) * np.sqrt(self.t)
        return np.append(wiener_process, -wiener_process)

    def get_gamma_process(self, nu=1):
        """
        Simulate Gamma process
        :param float nu: Gamma distribution's variance
        """
        gamma_process = np.random.gamma(self.t / nu, nu, size=self.n)
        return gamma_process

    def get_poisson_process(self, lam=1):
        """
        Simulate Poisson process
        :param float lam: Intensity of jumps
        """
        poisson_process = np.random.poisson(lam * self.t, size=self.n)
        return poisson_process

    def risk_neutralize(self, paths):
        """Risk neutralize simulations by mean correction"""
        return self.s * paths * np.exp(self.r * self.t) / np.mean(paths)


class GeometricBrownianMotion(StochasticProcess):
    """Simulate Geometric Brownian Motion using the explicit solution"""

    def __init__(self, s, r, vol, t, carry, n):
        super().__init__(s, r, vol, t, carry, n)

    def simulate(self):
        drift_component = (self.carry - 0.5 * self.vol**2) * self.t
        diffusive_component = self.vol * self.get_wiener_process()
        simulated_paths = self.s * np.exp(drift_component + diffusive_component)
        return simulated_paths


class VarianceGammaProcess(StochasticProcess):
    """Simulate Variance Gamma (VG) Process via Euler-Maruyama"""

    def __init__(self, s, r, vol, t, carry, n, params):
        """
        :param list params: List of process' parameters

        # Euler-Maruyama
        X_t = S_0 * exp(r * t + mu * dG + vol * sqrt(dG) * dW)

        with dG as Gamma increments and dW as Normal increments

        theta: Drift parameter
        nu: Variance parameter
        """
        super().__init__(s, r, vol, t, carry, n)
        try:
            self.theta = float(params[0])
            self.nu = float(params[1])
        except TypeError:
            self.logger.error("Missing process' parameters. Insert: Theta and Nu")
            sys.exit(1)

    def simulate(self):
        gamma_process = self.get_gamma_process(self.nu)
        normal_observations = np.random.randn(self.n)
        drift_component = self.theta * gamma_process
        diffusive_component = self.vol * normal_observations * np.sqrt(gamma_process)
        simulated_paths = self.s * np.exp(self.r * self.t + drift_component + diffusive_component)
        return self.risk_neutralize(simulated_paths)


class MertonJumpProcess(StochasticProcess):
    """Simulate Merton Jump Diffusion Process via Euler-Maruyama"""

    def __init__(self, s, r, vol, t, carry, n, params):
        """
        :param list params: List of process' parameters

        # Euler-Maruyama
        X_t = S_0 * exp(r * t + mu * t + vol * dW + dJ)

        with dW as a Wiener process and dJ as a compensated Poisson process normally distributed

        mu: Drift parameter
        lambda: Intensity of jumps
        mu_poisson: Mean for jumps' normal distribution
        vol_poisson: Standard deviation for jumps' distribution
        """
        super().__init__(s, r, vol, t, carry, n)
        try:
            self.mu = float(params[0])
            self.lambda_poisson = float(params[1])
            self.mu_poisson = float(params[2])
            self.vol_poisson = float(params[3])
        except TypeError:
            self.logger.error("Missing process' parameters")
            sys.exit(1)

    def simulate(self):
        poisson_process = self.get_poisson_process(self.lambda_poisson)
        wiener_process = self.get_wiener_process()
        drift_component = self.mu * self.t
        diffusive_component = self.vol * wiener_process
        jump_component = (
            np.random.normal(self.mu_poisson, self.vol_poisson, self.n) * poisson_process
        )
        simulated_paths = self.s * np.exp(
            self.r * self.t + drift_component + diffusive_component + jump_component
        )
        return self.risk_neutralize(simulated_paths)
