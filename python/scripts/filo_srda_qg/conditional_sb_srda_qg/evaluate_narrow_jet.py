from scripts.filo_srda_qg.sb_prior_retrain_qg.evaluate_narrow_jet import (
    parse_args,
    run_cycle,
)


if __name__ == "__main__":
    args = parse_args()
    if args.mode != "sr_only":
        raise ValueError("Experiment 072 excludes LETKF; use --mode sr_only")
    raise SystemExit(run_cycle(args))
