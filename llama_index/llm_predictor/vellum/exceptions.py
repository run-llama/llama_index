class VellumException(Exception):
    pass


class VellumApiError(VellumException):
    pass


class VellumGenerateException(VellumApiError):
    pass
