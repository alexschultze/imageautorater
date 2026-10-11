"""imageautorater: photo quality rating, keywords, face naming and a ranking gallery."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("imageautorater")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"
