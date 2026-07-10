from workflows.context.serializers import (
    BaseSerializer,  # noqa
    JsonSerializer,  # noqa
    MsgPackSerializer,
)

# provided for backward compatibility
JsonPickleSerializer = MsgPackSerializer