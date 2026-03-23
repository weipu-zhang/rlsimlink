from .dmlab import DMLab


def build_single_dmlab_env(env_name, **kwargs):
    """Construct a single DMLab environment instance."""
    return DMLab(level=env_name, **kwargs)
