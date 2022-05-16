import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.stochastic_processes import (
    GeometricBrownianMotion,
    VarianceGammaProcess,
    MertonJumpProcess,
)

# Set plot style
plt.style.use("seaborn-dark")
for param in ["figure.facecolor", "axes.facecolor", "savefig.facecolor"]:
    plt.rcParams[param] = "#212946"
for param in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
    plt.rcParams[param] = "0.9"


class PricingMethod:
    """
    Parent class for all the pricing methods
    """

    def __init__(self, product, s, f, k, r, rf, q, u, dividend, dividend_dates, vol, t, b, i):
        """
        :param str product: Selected product to price
        :param float s: Spot price
        :param float f: Forward price
        :param float k: Strike price
        :param float r: Risk-free (domestic) rate
        :param float rf: Risk-free (foreign) rate
        :param float q: Dividend yield - annual %
        :param float u: Storage cost - annual %
        :param list dividend: List of dividends, e.g. [0.5$, 0.5$, 0.5$]
        :param list dividend_dates: List of dividends dates, e.g. semi-annual [0.5, 1, 1.5]
        :param float vol: Log-normal volatility
        :param list t: Time to expiration
        :param float b: Bond price
        :param float i: PV of bond's income
        """
        self.logger = logging.getLogger(__name__)
        self.product = product
        self.s = s
        self.f = f
        self.k = k
        self.r = r
        self.rf = rf
        self.q = q
        self.u = u
        self.dividend = dividend
        self.dividend_dates = dividend_dates
        self.vol = vol
        self.b = b
        self.i = i

        # Convert time to expiration in years
        try:
            # Years
            if len(t) == 1:
                self.t = float(t[0])
            # Months
            elif t[1] in ["m", "M", "months"]:
                self.t = float(t[0]) / 12
            # Weeks
            elif t[1] in ["w", "W", "weeks"]:
                self.t = float(t[0]) / 52
            # Days
            else:
                self.t = float(t[0]) / 252

        except TypeError:
            self.logger.error("Missing time to expiration. Please insert it with -t")
            sys.exit(1)

        # Compute price for bond options
        if self.product in ["BC", "BP"]:
            self.f = (self.b - self.i) * np.exp(self.r * self.t)

        # Calculate cost of carry
        self.carry = (r - rf - q + u) if self.f is None else 0

        # Compute price used to calculate option value
        try:
            self.Price = (self.s * np.exp(self.carry * self.t)) if self.f is None else self.f
        except TypeError:
            self.logger.error("Missing S or F")
            sys.exit(1)

        # Get price net of dividends
        if self.dividend and f is None:
            for index, dividend in enumerate(self.dividend):
                self.Price = self.Price - np.exp(self.carry * self.t) * (
                    float(dividend) * np.exp(-self.r * float(self.dividend_dates[index]))
                )


class BSOption(PricingMethod):
    """
    Price options using Black-Scholes model
    """

    def __init__(self, product, s, f, k, r, rf, q, u, dividend, dividend_dates, vol, t, b, i):
        """
        :param str product: Selected product to price
        :param float s: Spot price
        :param float f: Forward price
        :param float k: Strike price
        :param float r: Risk-free (domestic) rate
        :param float rf: Risk-free (foreign) rate
        :param float q: Dividend yield - annual %
        :param float u: Storage cost - annual %
        :param list dividend: List of dividends, e.g. [0.5$, 0.5$, 0.5$]
        :param list dividend_dates: List of dividends dates, e.g. semi-annual [0.5, 1, 1.5]
        :param float vol: Log-normal volatility
        :param list t: Time to expiration
        :param float b: Bond price
        :param float i: PV of bond's income
        """
        super().__init__(product, s, f, k, r, rf, q, u, dividend, dividend_dates, vol, t, b, i)
        self.logger.info("Pricing using Black-Scholes")

        # Calculate d1 and d2
        try:
            self.d1 = (np.log(self.Price / self.k) + ((self.vol**2) * self.t / 2)) / (
                self.vol * np.sqrt(self.t)
            )

            self.d2 = self.d1 - (self.vol * np.sqrt(self.t))

        except TypeError:
            self.logger.error("Missing one of the following variables: S or F, K, r, vol, T")
            sys.exit(1)

    def get_price(self):
        if self.product == "EUC" or self.product == "BC":
            return np.exp(-self.r * self.t) * (
                (self.Price * norm.cdf(self.d1)) - (self.k * norm.cdf(self.d2))
            )

        elif self.product == "EUP" or self.product == "BP":
            return np.exp(-self.r * self.t) * (
                (self.k * norm.cdf(-self.d2)) - (self.Price * norm.cdf(-self.d1))
            )

        elif self.product == "FSTYLEC":
            return (self.Price * norm.cdf(self.d1)) - (self.k * norm.cdf(self.d2))

        elif self.product == "FSTYLEP":
            return (self.k * norm.cdf(-self.d2)) - (self.Price * norm.cdf(-self.d1))

        else:
            self.logger.error("Derivative security not found!")
            sys.exit(1)

    def get_theta(self):
        if self.product == "EUC":
            return np.exp(-self.r * self.t) * (
                -(self.Price * norm.pdf(self.d1) * self.vol / (2 * np.sqrt(self.t)))
                - ((self.carry - self.r) * self.Price * norm.cdf(self.d1))
                - (self.r * self.k * norm.cdf(self.d2))
            )

        elif self.product == "EUP":
            return np.exp(-self.r * self.t) * (
                -(self.Price * norm.pdf(self.d1) * self.vol / (2 * np.sqrt(self.t)))
                + ((self.carry - self.r) * self.Price * norm.cdf(-self.d1))
                + (self.r * self.k * norm.cdf(-self.d2))
            )

        else:
            self.logger.error("Greeks not available for selected instrument")
            sys.exit(1)

    def get_delta(self):
        if self.product == "EUC":
            return norm.cdf(self.d1) * np.exp((self.carry - self.r) * self.t)

        elif self.product == "EUP":
            return (norm.cdf(self.d1) - 1) * np.exp((self.carry - self.r) * self.t)

    def get_gamma(self):
        return (
            np.exp((self.carry - self.r) * self.t)
            * norm.pdf(self.d1)
            / (self.Price / np.exp(self.carry * self.t) * self.vol * np.sqrt(self.t))
        )

    def get_vega(self):
        return np.exp((self.carry - self.r) * self.t) * (
            norm.pdf(self.d1) * (self.Price / np.exp(self.carry * self.t)) * np.sqrt(self.t)
        )

    def get_rho(self):
        if self.product == "EUC":
            return self.k * np.exp(-self.r * self.t) * self.t * norm.cdf(self.d2)

        elif self.product == "EUP":
            return -self.k * np.exp(-self.r * self.t) * self.t * norm.cdf(-self.d2)


class BachelierOption(PricingMethod):
    """
    Price options using Bachelier model
    """

    def __init__(
        self, product, s, f, k, r, rf, q, u, dividend, dividend_dates, vol, nvol, t, b, i
    ):
        """
        :param str product: Selected product to price
        :param float s: Spot price
        :param float f: Forward price
        :param float k: Strike price
        :param float r: Risk-free (domestic) rate
        :param float rf: Risk-free (foreign) rate
        :param float q: Dividend yield - annual %
        :param float u: Storage cost - annual %
        :param list dividend: List of dividends, e.g. [0.5$, 0.5$, 0.5$]
        :param list dividend_dates: List of dividends dates, e.g. semi-annual [0.5, 1, 1.5]
        :param float vol: Log-normal volatility
        :param list t: Time to expiration
        :param float b: Bond price
        :param float i: PV of bond's income
        :param float nvol: Normal volatility
        """
        super().__init__(product, s, f, k, r, rf, q, u, dividend, dividend_dates, vol, t, b, i)
        self.logger.info("Pricing using Bachelier")

        # Normal volatility = log-normal volatility * F
        self.vol = nvol if nvol is not None else self.Price * self.vol

        try:
            # Calculate d_N
            self.d = (self.Price - self.k) / (self.vol * np.sqrt(self.t))

        except TypeError:
            self.logger.error("Missing one of the following variables: S or F, K, r, vol, T")
            sys.exit(1)

    def get_price(self):
        if self.product == "EUC" or self.product == "BC":
            return np.exp(-self.r * self.t) * (
                (self.Price - self.k) * norm.cdf(self.d)
                + (self.vol * np.sqrt(self.t) * norm.pdf(self.d))
            )

        elif self.product == "EUP" or self.product == "BP":
            return np.exp(-self.r * self.t) * (
                (self.k - self.Price) * norm.cdf(-self.d)
                + (self.vol * np.sqrt(self.t) * norm.pdf(self.d))
            )

        elif self.product == "FSTYLEC":
            return (self.Price - self.k) * norm.cdf(self.d) + (
                self.vol * np.sqrt(self.t) * norm.pdf(self.d)
            )

        elif self.product == "FSTYLEP":
            return (self.k - self.Price) * norm.cdf(-self.d) + (
                self.vol * np.sqrt(self.t) * norm.pdf(self.d)
            )

        else:
            self.logger.error("Derivative security not found!")
            sys.exit(1)


class BinomialTreeOption(PricingMethod):
    """
    Price options using Binomial trees
    """

    def __init__(
        self, product, s, f, k, r, rf, q, u, dividend, dividend_dates, vol, t, b, i, steps
    ):
        """
        :param str product: Selected product to price
        :param float s: Spot price
        :param float f: Forward price
        :param float k: Strike price
        :param float r: Risk-free (domestic) rate
        :param float rf: Risk-free (foreign) rate
        :param float q: Dividend yield - annual %
        :param float u: Storage cost - annual %
        :param list dividend: List of dividends, e.g. [0.5$, 0.5$, 0.5$]
        :param list dividend_dates: List of dividends dates, e.g. semi-annual [0.5, 1, 1.5]
        :param float vol: Log-normal volatility
        :param list t: Time to expiration
        :param float b: Bond price
        :param float i: PV of bond's income
        :param int steps: Number of steps in the binomial tree
        """
        super().__init__(product, s, f, k, r, rf, q, u, dividend, dividend_dates, vol, t, b, i)
        self.logger.info("Pricing using Binomial Tree")
        self.steps = steps

        # Initialize tree
        self.tree = np.full([2, self.steps + 1, self.steps + 1], np.nan)

        # Calculate the time period for each step
        self.dt = self.t / self.steps

        # Compute self.tree's parameters
        self.u = np.exp(self.vol * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.a = np.exp(self.carry * self.dt)
        self.p = (self.a - self.d) / (self.u - self.d)

        # Compute price used to calculate option value
        self.Price = self.s if self.f is None else self.f

    def get_price(self):
        """
        To price the option we'll use a 3-dimensional numpy array:
            Dimension 1: 0=Asset prices 1=Option prices
            Dimension 2: Step, hence the columns in a regular tree
            Dimension 3: State, hence the rows in a regular tree

        See Chp. 21 of Options, Futures, and Other Derivatives by John C. Hull for explanation
        """
        # Create tree of asset prices
        for i in range(self.steps + 1):
            for j in range(i + 1):
                self.tree[0, i, j] = np.array([self.Price * (self.u**j) * (self.d ** (i - j))])

        # Compute option payoff at expiration
        if self.product == "EUC" or self.product == "USC":
            self.tree[1, self.steps] = np.maximum(0, self.tree[0, self.steps] - self.k)
        elif self.product == "EUP" or self.product == "USP":
            self.tree[1, self.steps] = np.maximum(0, self.k - self.tree[0, self.steps])

        # Calculate tree of option prices
        for i in range(self.steps, 0, -1):
            for j in range(i):
                # When pricing american options we have to check if it's optimal to exercise early
                if self.product == "USC" or self.product == "USP":
                    execution_value = (
                        np.maximum(0, self.tree[0, i - 1, j] - self.k)
                        if self.product == "USC"
                        else np.maximum(0, self.k - self.tree[0, i - 1, j])
                    )
                    continuation_value = np.exp(-self.r * self.dt) * (
                        self.p * self.tree[1, i, j + 1] + (1 - self.p) * self.tree[1, i, j]
                    )

                    self.tree[1, i - 1, j] = (
                        execution_value
                        if execution_value > continuation_value
                        else continuation_value
                    )

                # If the option is not american we don't have to check for early exercise
                else:
                    self.tree[1, i - 1, j] = np.exp(-self.r * self.dt) * (
                        self.p * self.tree[1, i, j + 1] + (1 - self.p) * self.tree[1, i, j]
                    )

        # Return initial option price
        return self.tree[1, 0, 0]

    def print_tree(self):
        for i in range(self.steps, -1, -1):
            for j in range(i):
                # Connect price at t-1 with prices at t+1
                plt.plot(
                    [i, i - 1],
                    [self.tree[0, i, j + 1], self.tree[0, i - 1, j]],
                    "#FE53BB",
                    alpha=0.5,
                )
                plt.plot(
                    [i, i - 1],
                    [self.tree[0, i, j], self.tree[0, i - 1, j]],
                    "#FE53BB",
                    alpha=0.5,
                )

                # Annotate option prices
                plt.annotate(
                    round(self.tree[1, i, j + 1], 2),
                    (i, self.tree[0, i, j + 1]),
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontweight="heavy",
                )
                plt.annotate(
                    round(self.tree[1, i, j], 2),
                    (i, self.tree[0, i, j]),
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontweight="heavy",
                )

                # Annotate asset prices
                plt.annotate(
                    round(self.tree[0, i, j + 1], 2),
                    (i, self.tree[0, i, j + 1]),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontweight="heavy",
                )
                plt.annotate(
                    round(self.tree[0, i, j], 2),
                    (i, self.tree[0, i, j]),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontweight="heavy",
                )

        plt.xticks(
            np.arange(0, self.steps + 1, 1),
            fontweight="bold",
        )
        plt.yticks(
            np.linspace(min(self.tree[0, self.steps]), max(self.tree[0, self.steps]), 10),
            fontweight="heavy",
        )
        plt.grid(color="#2A3459")
        plt.show()


class MonteCarloSimulation(PricingMethod):
    """
    Price options using Monte Carlo simulation under different stochastic processes
    """

    def __init__(
        self, product, s, f, k, r, rf, q, u, dividend, dividend_dates, vol, t, b, i, n, process, params
    ):
        """
        :param str product: Selected product to price
        :param float s: Spot price
        :param float f: Forward price
        :param float k: Strike price
        :param float r: Risk-free (domestic) rate
        :param float rf: Risk-free (foreign) rate
        :param float q: Dividend yield - annual %
        :param float u: Storage cost - annual %
        :param list dividend: List of dividends, e.g. [0.5$, 0.5$, 0.5$]
        :param list dividend_dates: List of dividends dates, e.g. semi-annual [0.5, 1, 1.5]
        :param float vol: Log-normal volatility
        :param list t: Time to expiration
        :param float b: Bond price
        :param float i: PV of bond's income
        :param int n: Number of simulated paths
        :param str process: Selected underlying process
        :param list params: Underlying process' parameters
        """
        super().__init__(product, s, f, k, r, rf, q, u, dividend, dividend_dates, vol, t, b, i)
        self.logger.info("Pricing using Monte Carlo Simulation")
        self.n = n
        self.process = process
        self.params = params

    def get_price(self):
        # Select stochastic process here
        if self.process == "GBM":
            process = GeometricBrownianMotion(self.s, self.r, self.vol, self.t, self.carry, self.n)
            self.logger.info("Pricing under Geometric Brownian Motion")
        elif self.process == "VG":
            process = VarianceGammaProcess(
                self.s, self.r, self.vol, self.t, self.carry, self.n, self.params
            )
            self.logger.info("Pricing under Variance Gamma Process")
        elif self.process == "MJ":
            process = MertonJumpProcess(
                self.s, self.r, self.vol, self.t, self.carry, self.n, self.params
            )
            self.logger.info("Pricing under Merton Jump Process")
        else:
            self.logger.error("Underlying process not supported. Select: GBM, VG or MJ")
            sys.exit(1)

        # Get simulated paths
        try:
            simulated_paths = process.simulate()

        except TypeError:
            self.logger.error("Missing one of the following variables: S, K, r, vol, T, n")
            sys.exit(1)

        # Compute payoff
        if self.product in ["EUC", "BC"]:
            payoff = np.where(simulated_paths - self.k < 0, 0, simulated_paths - self.k)
        else:
            payoff = np.where(self.k - simulated_paths < 0, 0, simulated_paths - self.k)

        # Discount payoff to get the option price
        return payoff.mean() * np.exp(-self.r * self.t)
