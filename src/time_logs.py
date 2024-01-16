import time

class TimerLog:
    """ A utility class for timing operations, acting as a stopwatch. """

    def __init__(self):
        """ Initializes the timer to the current system time. """
        self.start_time = time.time()

    def start(self):
        """ Resets the start time to the current system time. """
        self.start_time = time.time()

    def get_elapsed(self):
        """ Returns the elapsed time in seconds since the last reset. """
        return time.time() - self.start_time
