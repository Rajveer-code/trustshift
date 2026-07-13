"""Lending adapter entry point (alias to train.main).

Lending trains its own models (unlike clinical/nlp), so the adapter and trainer are the same
step. `python -m domains.lending.adapter` and `python -m domains.lending.train` are equivalent.
"""
from domains.lending.train import main

if __name__ == "__main__":
    main()
