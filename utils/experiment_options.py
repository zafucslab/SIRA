"""Read current experiment options and accept their historical names."""

import os


def experiment_option(name, default=None):
    if name in os.environ:
        return os.environ[name]

    if name == "SIRA_FREEZE_BASE":
        return os.environ.get("CHSRF_FREEZE_BASE", default)
    if name == "SIRA_TIME_LOSS":
        return os.environ.get("RENF_TIME_LOSS", default)

    prefix, _, suffix = name.partition("_")
    legacy_prefix = {"DARA": "CHSRF", "PCFB": "UNCLE"}.get(prefix)
    if legacy_prefix is not None:
        return os.environ.get(f"{legacy_prefix}_{suffix}", default)
    return default
