import argparse
import logging
import sys

from src.pricers import (
    BSOption,
    BachelierOption,
    BinomialTreeOption,
    MonteCarloSimulation,
)

parser = argparse.ArgumentParser(
    description="Option pricer", formatter_class=argparse.RawTextHelpFormatter
)

# General arguments
general = parser.add_argument_group("General")
general.add_argument(
    "-m",
    "--mode",
    help="Select pricing method: \n"
    "    * Black-Scholes: BS \n"
    "    * Bachelier: BA \n"
    "    * Binomial Tree: BT \n"
    "    * Monte Carlo Simulation: MC",
    default="BS",
    choices=["BS", "BA", "BT", "MC"],
)
general.add_argument(
    "-p",
    "--product",
    help="Select product to price: \n"
    "    * EU options: EUC (call), EUP (put) \n"
    "    * American options: USC (call), USP (put) \n"
    "    * Futures-style options: FSTYLEC (call), FSTYLEP (put) \n"
    "    * Bond options: BC (call), BP (put)",
    required=True,
    choices=["EUC", "EUP", "USC", "USP", "FSTYLEC", "FSTYLEP", "BC", "BP"],
)
general.add_argument(
    "-s",
    dest="spot",
    help="Spot price | S_0",
    type=float,
)
general.add_argument(
    "-f",
    dest="forward",
    default=None,
    help="Forward price | F",
    type=float,
)
general.add_argument(
    "-k",
    dest="strike",
    help="Strike price | K",
    type=float,
    required=True,
)
general.add_argument(
    "-r",
    dest="riskless_rate",
    help="Annualized (domestic) risk-free rate | r",
    type=float,
    required=True,
)
general.add_argument(
    "-rf",
    dest="foreign_rate",
    help="Annualized (foreign) risk-free rate | rf",
    default=0,
    type=float,
)
general.add_argument(
    "-u",
    dest="storage",
    help="Annualized storage cost | u",
    default=0,
    type=float,
)
general.add_argument(
    "-q",
    dest="dividend_yield",
    help="Annualized dividend yield | q",
    default=0,
    type=float,
)
general.add_argument(
    "-d",
    "--dividends",
    dest="dividends",
    help="List of dividends \ne.g. 3x 0.5$ dividends: 0.5 0.5 0.5",
    nargs="+",
)
general.add_argument(
    "-dt",
    "--dividend_times",
    dest="dividend_times",
    help="List of dividends times \ne.g. semi-annual dividends: 0.5 1 1.5",
    nargs="+",
)
general.add_argument(
    "-t",
    dest="time_to_expiration",
    help="Time to expiration | T",
    nargs="+",
    required=True,
)
general.add_argument(
    "-vol",
    dest="volatility",
    help="Annualized log-normal volatility",
    type=float,
)
general.add_argument(
    "-nvol",
    dest="normal_volatility",
    help=f"Annualized normal volatility",
    type=float,
)
general.add_argument(
    "-b",
    dest="bond_price",
    help="Current bond cash price | B_0",
    type=float,
)
general.add_argument(
    "-i",
    dest="income_pv",
    help="PV of bond's income",
    type=float,
)

# Arguments for Binomial Trees
tree = parser.add_argument_group("Binomial Trees")
tree.add_argument(
    "-steps",
    help="Number of steps for binomial tree",
    type=int,
    default=4,
)
tree.add_argument(
    "-print",
    dest="print_tree",
    help="Print binomial tree",
    action="store_true",
)

# Arguments for Black-Scholes
bs = parser.add_argument_group("Black-Scholes")
bs.add_argument(
    "-greeks",
    help="Print greeks values",
    action="store_true",
)

# Arguments for Monte Carlo Simulation
mc = parser.add_argument_group("Monte Carlo simulation")
mc.add_argument(
    "-n",
    dest="trials",
    help="Number of paths to simulate",
    type=int,
)
mc.add_argument(
    "-process",
    help="Select underlying process: \n"
    "    * Geometric Brownian Motion: GBM \n"
    "    * Variance-Gamma: VG \n"
    "    * Merton-Jump: MJ",
    default="GBM",
    choices=["GBM", "VG", "MJ"],
)
mc.add_argument(
    "-params",
    help="Insert process parameters ",
    nargs="+",
)
args = parser.parse_args()

# Logger
logger = logging.getLogger()
logger_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # Black-Scholes
    if args.mode == "BS":
        if args.product in ["USC", "USP"]:
            logger.error("Black-Scholes not available for American options")
            sys.exit(1)

        option = BSOption(
            args.product,
            args.spot,
            args.forward,
            args.strike,
            args.riskless_rate,
            args.foreign_rate,
            args.dividend_yield,
            args.storage,
            args.dividends,
            args.dividend_times,
            args.volatility,
            args.time_to_expiration,
            args.bond_price,
            args.income_pv,
        )

    # Bachelier
    elif args.mode == "BA":
        if args.product in ["USC", "USP"]:
            logger.error("Bachelier not available for American options")
            sys.exit(1)

        option = BachelierOption(
            args.product,
            args.spot,
            args.forward,
            args.strike,
            args.riskless_rate,
            args.foreign_rate,
            args.dividend_yield,
            args.storage,
            args.dividends,
            args.dividend_times,
            args.volatility,
            args.normal_volatility,
            args.time_to_expiration,
            args.bond_price,
            args.income_pv,
        )

    # Binomial Tree
    elif args.mode == "BT":
        if args.product in ["USC", "USP", "EUC", "EUP"]:
            option = BinomialTreeOption(
                args.product,
                args.spot,
                args.forward,
                args.strike,
                args.riskless_rate,
                args.foreign_rate,
                args.dividend_yield,
                args.storage,
                args.dividends,
                args.dividend_times,
                args.volatility,
                args.time_to_expiration,
                args.bond_price,
                args.income_pv,
                args.steps,
            )
        else:
            logger.error(
                f"Binomial tree not available for {args.product}. Select USC, USP, EUC or EUP"
            )
            sys.exit(1)

    # Monte Carlo Simulation
    elif args.mode == "MC":
        if args.product in ["EUC", "EUP"]:
            option = MonteCarloSimulation(
                args.product,
                args.spot,
                args.forward,
                args.strike,
                args.riskless_rate,
                args.foreign_rate,
                args.dividend_yield,
                args.storage,
                args.dividends,
                args.dividend_times,
                args.volatility,
                args.time_to_expiration,
                args.bond_price,
                args.income_pv,
                args.trials,
                args.process,
                args.params,
            )
        else:
            logger.error(f"Monte Carlo not available for {args.product}. Select EUC or EUP")
            sys.exit(1)

    else:
        logger.error(f"-m {args.mode} not available. Select -m BS, BA, BT or MC")
        sys.exit(1)

    logger.info(f"The option price is: {round(option.get_price(), 5)}")

    if args.greeks:
        if args.mode == "BS":
            logger.info(
                "The Greeks are: "
                f"Theta {round(option.get_theta(), 5)} | "
                f"Delta {round(option.get_delta(), 5)} | "
                f"Gamma {round(option.get_gamma(), 5)} | "
                f"Vega {round(option.get_vega(), 5)} | "
                f"Rho {round(option.get_rho(), 5)}"
            )
        else:
            logger.error(f"Greeks not available for {args.mode}")
            sys.exit(1)

    if args.print_tree:
        if args.mode == "BT":
            option.print_tree()
        else:
            logger.error(f"Cannot print binomial tree using {args.mode} mode")
            sys.exit(1)
