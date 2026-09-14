# -*- coding: utf-8 -*-
"""Define monthly training, validation and test origins and target intervals.

Each origin t uses X = dat[t-P:t] and Y = dat[t:t+out]. The legacy policy
partitions origins without requiring disjoint target intervals. withheld
reserves final test and validation blocks; withheld_matched preserves legacy
test origins while separating training and validation targets. Both withheld
policies assert disjoint target unions by default.

The public pipeline selects withheld_matched. This standalone module retains
legacy as the build() function default for historical callers. It uses only
the standard library."""

POLICIES = ("legacy", "withheld", "withheld_matched")


class SplitError(ValueError):
    pass


def _targets(starts, out):
    """Half-open union [lo, hi) of the target blocks of these origins."""
    if not starts:
        return None
    return (min(starts), max(starts) + out)


def _overlap(a, b):
    if a is None or b is None:
        return 0
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def build(n, P, h, out, policy="legacy", num_eval=7,
          test_reserve=24, valid_span=24, strict=None):
    """Build origin lists and normalization boundaries for a monthly panel.
    
    Args:
        n: Panel length in months.
        P: Input-window length.
        h: Forecast offset; 1 in this project.
        out: Target-block length.
        policy: legacy, withheld or withheld_matched.
        num_eval: Test-origin count for legacy and withheld_matched; also the
            validation-origin count for legacy.
        test_reserve: Test-block length for withheld.
        valid_span: Validation-block length for either withheld policy.
        strict: Check nonempty splits and disjoint target unions; defaults to
            True for withheld policies and False for legacy.
    
    Return origin lists, train_end, target unions and overlap counts."""
    if policy not in POLICIES:
        raise SplitError("unknown policy %r; expected one of %s" % (policy, POLICIES))
    if strict is None:
        strict = policy.startswith("withheld")

    first = P + h - 1                      # earliest origin with a full input window

    if policy == "legacy":
        test_starts = list(range(n - out - num_eval + 1, n - out + 1))
        valid_starts = list(range(n - out - 2 * num_eval + 1, n - out - num_eval + 1))
        train_starts = list(range(first, n - out - 2 * num_eval + 1))
        train_end = n - 2 * num_eval
        bounds = None
    elif policy == "withheld":
        t0 = n - test_reserve              # first test target month
        v0 = t0 - valid_span               # first validation target month
        test_starts = list(range(t0, n - out + 1))
        valid_starts = list(range(v0, t0 - out + 1))
        train_starts = list(range(first, v0 - out + 1))
        train_end = v0                     # normalisation may not see the held-out months
        bounds = (v0, t0)
    else:  # withheld_matched
        # Preserve legacy test origins while excluding their targets from training and validation.
        # Normalization ends at the revised training boundary.
        test_starts = list(range(n - out - num_eval + 1, n - out + 1))
        t0 = min(test_starts)              # first test target month
        v0 = t0 - valid_span
        valid_starts = list(range(v0, t0 - out + 1))
        train_starts = list(range(first, v0 - out + 1))
        train_end = v0
        bounds = (v0, t0)

    tr, va, te = (_targets(s, out) for s in (train_starts, valid_starts, test_starts))

    if strict:
        for name, empty in (("train", not train_starts), ("validation", not valid_starts),
                            ("test", not test_starts)):
            if empty:
                raise SplitError(
                    "policy %r leaves no %s origin at horizon %d on a %d-month panel "
                    "(test_reserve=%d, valid_span=%d, P=%d). A block of %d target months plus a "
                    "full input window does not fit; use a longer reserve, a shorter input "
                    "window, or report this horizon as not evaluable."
                    % (policy, name, out, n, test_reserve, valid_span, P, out))
        for a, b, la, lb in ((tr, te, "train", "test"), (va, te, "validation", "test"),
                             (tr, va, "train", "validation")):
            k = _overlap(a, b)
            if k:
                raise SplitError("%s and %s target blocks overlap by %d months "
                                 "under policy %r" % (la, lb, k, policy))
        if train_end > (bounds[0] if bounds else n):
            raise SplitError("normalisation boundary %d reaches into the held-out period"
                             % train_end)

    return {
        "policy": policy,
        "train_starts": train_starts,
        "valid_starts": valid_starts,
        "test_starts": test_starts,
        "train_end": train_end,
        "targets": {"train": tr, "valid": va, "test": te},
        "overlap_months": {"train_test": _overlap(tr, te),
                           "valid_test": _overlap(va, te),
                           "train_valid": _overlap(tr, va)},
        "params": {"n": n, "P": P, "h": h, "out": out, "num_eval": num_eval,
                   "test_reserve": test_reserve, "valid_span": valid_span},
    }


def month(i, start_year=2014, start_month=1):
    k = (start_month - 1) + i
    return "%04d-%02d" % (start_year + k // 12, k % 12 + 1)


def report(plan, start_year=2014, start_month=1):
    """One block of text naming the actual months, printed at load time so a
    run's log always records which partition produced it."""
    m = lambda i: month(i, start_year, start_month)
    p, t = plan["params"], plan["targets"]
    rng = lambda s: "-" if s is None else "%s..%s" % (m(s[0]), m(s[1] - 1))
    ov = plan["overlap_months"]
    lines = [
        "[SPLIT] policy=%s  horizon=%d  window=%d  panel=%d months"
        % (plan["policy"], p["out"], p["P"], p["n"]),
        "[SPLIT]   origins   train %d | valid %d | test %d"
        % (len(plan["train_starts"]), len(plan["valid_starts"]), len(plan["test_starts"])),
        "[SPLIT]   targets   train %s" % rng(t["train"]),
        "[SPLIT]             valid %s" % rng(t["valid"]),
        "[SPLIT]             test  %s" % rng(t["test"]),
        "[SPLIT]   normalisation uses months up to %s (index %d)"
        % (m(plan["train_end"] - 1), plan["train_end"]),
        "[SPLIT]   target overlap  train/test %d  valid/test %d  train/valid %d months"
        % (ov["train_test"], ov["valid_test"], ov["train_valid"]),
    ]
    if any(ov.values()):
        lines.append("[SPLIT]   WARNING: target blocks overlap across partitions; "
                     "test error from this run is not a withheld-period error.")
    return "\n".join(lines)


if __name__ == "__main__":
    for pol in POLICIES:
        for H in (6, 9, 12, 24, 36):
            try:
                print(report(build(144, H, 1, H, policy=pol)))
            except SplitError as e:
                print("[SPLIT] policy=%s  horizon=%d  REFUSED: %s" % (pol, H, e))
            print()
