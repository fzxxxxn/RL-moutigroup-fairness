from typing import Dict, Callable

_MODELS: Dict[str, Callable] = {}
_METRICS: Dict[str, Callable] = {}
_REWARDS: Dict[str, Callable] = {}

def register_model(name: str):
    def deco(fn):
        _MODELS[name] = fn
        return fn
    return deco

def register_metric(name: str):
    def deco(fn):
        _METRICS[name] = fn
        return fn
    return deco

def register_reward(name: str):
    def deco(fn):
        _REWARDS[name] = fn
        return fn
    return deco

def get_model(name: str):
    return _MODELS[name]

def get_metric(name: str):
    return _METRICS[name]

def get_reward(name: str):
    return _REWARDS[name]
