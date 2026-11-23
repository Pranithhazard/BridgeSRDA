from scripts.filo_srda_qg.sb_prior_retrain_qg.evaluate_narrow_jet import (
    parse_args,
    run_cycle,
)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode != "sr_only":
        raise ValueError("Experiment 073A performs direct SB-SRDA; use --mode sr_only")
    raise SystemExit(run_cycle(parsed))
