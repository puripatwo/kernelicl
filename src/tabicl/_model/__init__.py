"""Private. Users should not import from this subpackage.

``TabICL`` is wrapped by :class:`tabicl.TabICLClassifier`,
:class:`tabicl.TabICLRegressor`, and :class:`tabicl.TabICLForecaster`.
``InferenceConfig`` is re-exported at :class:`tabicl.InferenceConfig`.
"""

from .tabicl import TabICL
from .inference_config import InferenceConfig
from .kernel_head import KernelHead, relative_perplexity
