import logging
import multiprocessing

from urban_planner.domain.geometries import Plot

logger = logging.getLogger(__name__)

class EnvironmentGenerator(multiprocessing.Process):
    """
    A dedicated process that continuously generates valid environment configurations
    and puts them into a queue for consumption by parallel training environments.
    """

    def __init__(self, queue: multiprocessing.Queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.stop_event = multiprocessing.Event()

    def run(self):
        """The main loop of the producer process."""
        logger.info(f"EnvironmentGenerator process started (PID: {self.pid}).")
        while not self.stop_event.is_set():
            try:
                plot = Plot.from_random_generation()
                self.queue.put(plot)
            except Exception as e:
                logger.error(f"Error during environment generation: {e}",
                             exc_info=True)

        logger.info(
            f"EnvironmentGenerator process (PID: {self.pid}) received stop signal and is shutting down.")

    def stop(self):
        """Signals the process to stop."""
        self.stop_event.set()