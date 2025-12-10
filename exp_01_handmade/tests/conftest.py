import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*mode.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision.transforms.functional")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
