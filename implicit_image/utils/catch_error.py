import logging
import sys
import traceback


def catch_error_decorator(func):
    """
    Super useful for debugging long slurm jobs
    :param func: function to execute
    :return:
    """

    def run_safely(*args, **kwargs):
        try:
            out = func(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
            logging.info(traceback.format_exc())
            sys.exit(1)
        return out

    return run_safely


@catch_error_decorator
def test(a, b, c):
    print(a, b, c)
    raise


if __name__ == "__main__":
    test(1, 2, c=3)
