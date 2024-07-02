#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-01-05 14:24:02
# @Last modified by: AlixBernard
# @Last modified time: 2023-02-26 08:04:27


from fluidml.version import version as __version__

__all__ = ["__version__"]


from fluidml import features
from fluidml import utils

__all__ += features.__all__
__all__ += utils.__all__
