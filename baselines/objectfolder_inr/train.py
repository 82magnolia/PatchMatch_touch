#!/usr/bin/env python3
"""Training-only convenience entry point for the ObjectFolder INR baseline."""

import sys

from run_baseline import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--train_only", "--train_if_missing"]
    main()
