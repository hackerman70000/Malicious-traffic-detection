from enum import Enum

ATTACK_TYPES = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]


class ModelType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
