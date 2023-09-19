

class ParametersTypeError(Exception):
    pass


class NotFittedError(Exception):
    pass


def augmented_isinstance(s, types):
    if types is None:
        return s is types
    elif isinstance(types, type):
        return isinstance(s, types)
    elif isinstance(types, tuple):
        without_none_types = tuple(filter(None, types))
        if len(without_none_types) == len(types):
            return isinstance(s, types)
        else:
            return isinstance(s, without_none_types) or (s is None)
