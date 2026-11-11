import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["INTENTCX_OFFLINE"] = "1"  # tests never hit the network
