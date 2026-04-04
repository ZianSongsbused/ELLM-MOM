import signal

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("Function call timed out!")

# 用法包装
def run_with_timeout(func, args=(), kwargs={}, timeout=5):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        result = None
    finally:
        signal.alarm(0)
    return result
