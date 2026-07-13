"""Security adapter entry point (alias to run_pipeline.main).

Security trains here (like lending), so adapter and pipeline are the same step.
`python -m domains.security.adapter` == `python -m domains.security.run_pipeline`.
"""
from domains.security.run_pipeline import main

if __name__ == "__main__":
    main()
