"""
Authors : inzapp

Github : https://github.com/inzapp/sbd
"""
class Logger:
    def __init__(self):
        pass

    @staticmethod
    def log(msg, header, end='\n'):
        msg_type = type(msg)
        assert msg_type in [str, list]
        if msg_type is str:
            print(f'{header} {msg}', end=end)
        else:
            for i, s in enumerate(msg):
                if i == 0:
                    print(f'{header} {s}')
                else:
                    print(f'    {s}')

    @staticmethod
    def info(msg, end='\n'):
        Logger.log(msg, header='\033[1;32m[INFO]\033[0m', end=end)

    @staticmethod
    def warn(msg, end='\n'):
        Logger.log(msg, header='\033[1;33m[WARNING]\033[0m', end=end)

    @staticmethod
    def error(msg, end='\n', callback=None):
        Logger.log(msg, header='\033[1;31m[ERROR]\033[0m', end=end)
        if callback:
            callback()
        exit(-1)

