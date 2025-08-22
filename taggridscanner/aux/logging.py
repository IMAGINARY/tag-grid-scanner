import logging


def configure_logging(verbosity):
    log_levels = [
        logging.CRITICAL + 10,  # 0
        logging.CRITICAL,  # 1
        logging.ERROR,  # 2
        logging.WARNING,  # 3
        logging.INFO,  # 4
        logging.DEBUG,  # 5
        logging.NOTSET,  # 6
    ]

    level = log_levels[min(max(0, verbosity), len(log_levels) - 1)]
    formatter = MultiLineFormatter(fmt="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(level)


class MultiLineFormatter(logging.Formatter):
    """Multi-line formatter."""

    def get_header_length(self, record):
        """Get the header length of a given record."""
        return len(
            super().format(
                logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg="",
                    args=(),
                    exc_info=None,
                )
            )
        )

    def format(self, record):
        """Format a record with added indentation."""
        indent = " " * self.get_header_length(record)
        head, *trailing = super().format(record).splitlines(True)
        return head + "".join(indent + line for line in trailing)
