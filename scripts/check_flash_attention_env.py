"""Project script entrypoint."""
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from unsloth_finetune.training.distributed.check_flash_attention_env import main

if __name__ == "__main__":
    main()

