

class UnifiedPrint:
    def __init__(self, logger, log_file_path='logs.log', silent=False):
        import logging
        import os

        if os.path.isfile(log_file_path):
            os.remove(log_file_path)

        self.logger = logger
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # 使用FileHandler输出到文件
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        if not silent:
            # 使用StreamHandler输出到屏幕
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)

            # 添加两个Handler
            self.logger.addHandler(ch)

    def info(self, s):
        self.logger.info(s)

    def debug(self, s):
        self.logger.debug(s)

    def error(self, s):
        self.logger.error(s)

    def warn(self, s):
        self.logger.warn(s)
