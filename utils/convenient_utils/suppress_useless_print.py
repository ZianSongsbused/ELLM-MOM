import contextlib
import os
import sys


# 有些底层实现会输出很多没用的内容（logger、c扩展等），不需要的时候全部屏蔽
@contextlib.contextmanager
def suppress_everything():
    """彻底屏蔽一切 stdout stderr（包括C扩展，logging，print，等等）"""
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(1)  # 保存原有stdout文件描述符
        old_stderr_fd = os.dup(2)  # 保存原有stderr文件描述符

        try:
            os.dup2(devnull.fileno(), 1)  # 重定向stdout到null
            os.dup2(devnull.fileno(), 2)  # 重定向stderr到null
            yield
        finally:
            os.dup2(old_stdout_fd, 1)  # 恢复原stdout
            os.dup2(old_stderr_fd, 2)  # 恢复原stderr
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

@contextlib.contextmanager
def suppress_console():
    """只屏蔽终端控制台输出，不影响文件写入"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
