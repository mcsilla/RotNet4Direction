#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""

from config.train import gcp_tpu_config

CONFIG_MAP = {
    'gcp': gcp_tpu_config.CONFIG,
}
