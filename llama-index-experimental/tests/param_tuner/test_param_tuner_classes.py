from llama_index.experimental.param_tuner import (
    AsyncParamTuner,
    BaseParamTuner,
    ParamTuner,
    RayTuneParamTuner,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in AsyncParamTuner.__mro__]
    assert BaseParamTuner.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ParamTuner.__mro__]
    assert BaseParamTuner.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in RayTuneParamTuner.__mro__]
    assert BaseParamTuner.__name__ in names_of_base_classes
