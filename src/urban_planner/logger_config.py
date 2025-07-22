import logging
import sys

def setup_logging():
    """
    Configures the root logger for the application.

    Sets up a basic configuration to stream logs of level INFO and higher
    to the standard output, with a clear and informative format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        stream=sys.stdout
    )