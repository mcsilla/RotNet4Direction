#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""


import config.streetview_config
import config.sample_config

CONFIG_MAP = {
    'sample_config': config.sample_config.CONFIG,
    'streetview_config': config.streetview_config.CONFIG,
}
