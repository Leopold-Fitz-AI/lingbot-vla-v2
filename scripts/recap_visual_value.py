#!/usr/bin/env python3
"""Train the visual-task RECAP value critic from recorded rollouts.

This is ``tasks/vla/train_recap_value.py`` with ``--encoder visual_task`` locked.
Flags match the shared trainer except ``--encoder``, which is not accepted.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.vla.train_recap_value import main as train_main  # noqa: E402
from tasks.vla.train_recap_value import parse_args  # noqa: E402


def main() -> None:
    args = parse_args(encoder_default="visual_task", lock_encoder=True)
    train_main(args)


if __name__ == "__main__":
    main()
