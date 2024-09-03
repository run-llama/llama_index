class WorkflowValidationError(Exception):
    pass


class WorkflowTimeoutError(Exception):
    pass


class WorkflowRuntimeError(Exception):
    pass


class WorkflowDone(Exception):
    pass
