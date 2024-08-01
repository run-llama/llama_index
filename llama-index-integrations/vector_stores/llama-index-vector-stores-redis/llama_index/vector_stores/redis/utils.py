from redisvl.query.filter import Tag, Num, Text


# Global constant defining field specifications
REDIS_LLAMA_FIELD_SPEC = {
    "tag": {
        "class": Tag,
        "operators": {
            "==": lambda f, v: f == v,
            "!=": lambda f, v: f != v,
            "in": lambda f, v: f == v,
            "nin": lambda f, v: f != v,
            "contains": lambda f, v: f == v,
        },
    },
    "numeric": {
        "class": Num,
        "operators": {
            "==": lambda f, v: f == v,
            "!=": lambda f, v: f != v,
            ">": lambda f, v: f > v,
            "<": lambda f, v: f < v,
            ">=": lambda f, v: f >= v,
            "<=": lambda f, v: f <= v,
        },
    },
    "text": {
        "class": Text,
        "operators": {
            "==": lambda f, v: f == v,
            "!=": lambda f, v: f != v,
            "text_match": lambda f, v: f % v,
        },
    },
}
