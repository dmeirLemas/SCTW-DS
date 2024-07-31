import sys
import threading


class ProgressBar:
    def __init__(
        self,
        total: int,
        program_name: str,
        bar_length=20,
        done_message="Finished!",
        update_interval=1,
        cost=None,
    ):
        self.total = total
        self.bar_length = bar_length
        self.done_message = done_message
        self.update_interval = update_interval
        self.current = 0
        self.lock = threading.Lock()

        print(f"\n\x1b[32m----Starting {program_name}----\x1b[0m\n")

        self.update(0, cost)

    def update(self, current: int, cost):
        with self.lock:
            self.current = current
            fraction = self.current / self.total

            arrow = int(fraction * self.bar_length) * "-" + ">"
            padding = (self.bar_length - len(arrow)) * " "

            if self.current == self.total:
                ending = f"\n\n{self.done_message}\x1b[0m\n\n"
            else:
                ending = "\r"

            completed = "\x1b[32m" if self.current == self.total else "\x1b[0m"

            print(
                f"{completed}Progress: [{arrow}{padding}] {fraction*100:.2f}% Cost: {cost}",
                end=ending,
            )
            sys.stdout.flush()

    def increment(self, step=1, cost=None):
        self.update(self.current + step, cost)
