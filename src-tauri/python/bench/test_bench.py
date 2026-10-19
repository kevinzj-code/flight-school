"""The Phase 1 gate: `bench/` must reproduce the earlier harness's published behaviors.

Every number asserted here is quoted from the earlier harness — `AUDIT.md`'s
p-value table, `NOTES.md`'s 28:7 label-ratio failure, the 0.82 leak threshold
that three separate files agreed on. If a value in here changes, the port
drifted from the thing it was ported from, and that is the whole point of the
file.

    python3 -m pytest src-tauri/python/bench/test_bench.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from bench import (
    mcnemar, mcnemar_floor, discordant,
    too_close, find_leaks, LEAK_THRESHOLD,
    label_ratio_problems, sequence_length_problems,
    contradiction_problems, duplicate_problems, audit_rows, subject_variety_problems,
    MAX_LABEL_RATIO,
)
from bench.corpus import estimate_tokens, CHARS_PER_TOKEN_DENSE
from bench.significance import verdict


# ---------------------------------------------------------------------------
# significance — AUDIT.md's own table


@pytest.mark.parametrize("n, expected", [(3, 0.250), (4, 0.125), (5, 0.0625), (6, 0.031)])
def test_mcnemar_floor_matches_the_published_table(n, expected):
    """The earlier harness's AUDIT.md publishes these four floors, and uses them to argue
    that six discordant pairs is the minimum for p < 0.05. If this drifts, the
    "is this bench even capable of proving anything" advice drifts with it.
    """
    assert mcnemar_floor(n) == pytest.approx(expected, abs=0.001)


def test_six_discordant_pairs_is_the_first_that_can_clear_significance():
    assert mcnemar_floor(5) > 0.05
    assert mcnemar_floor(6) < 0.05


def test_the_stage_5_headline_reproduces():
    """AUDIT.md: "5 fixed / 0 broken, p = 0.0625" — reported at the time as a
    result, and the audit's own conclusion was that it never cleared the bar.
    """
    a = [False] * 5 + [True] * 10
    b = [True] * 15
    fixed, broken, p = mcnemar(a, b)
    assert (fixed, broken) == (5, 0)
    assert p == pytest.approx(0.0625, abs=1e-6)
    assert "NOT significant" in verdict(fixed, broken, p)


def test_ten_fixed_none_broken_is_significant():
    """AUDIT.md: "v2 vs v1 on the adversarial bench: 10 fixed / 0 broken,
    p = 0.00195"."""
    a = [False] * 10 + [True] * 20
    b = [True] * 30
    fixed, broken, p = mcnemar(a, b)
    assert (fixed, broken) == (10, 0)
    assert p == pytest.approx(0.00195, abs=1e-5)


def test_seven_fixed_none_broken():
    """AUDIT.md: "v2 vs base on the withheld holdout: 7 fixed / 0 broken,
    p = 0.0156"."""
    _, _, p = mcnemar([False] * 7 + [True] * 5, [True] * 12)
    assert p == pytest.approx(0.0156, abs=1e-4)


def test_nine_fixed_none_broken():
    """AUDIT.md: "C over B: 9 fixed / 0 broken, p = 0.0039"."""
    _, _, p = mcnemar([False] * 9 + [True] * 5, [True] * 14)
    assert p == pytest.approx(0.0039, abs=1e-4)


def test_three_fixed_one_broken_is_directional_not_significant():
    """AUDIT.md: "3 fixed / 1 broken, p = 0.625 — directional, not significant"."""
    a = [False, False, False, True] + [True] * 10
    b = [True, True, True, False] + [True] * 10
    fixed, broken, p = mcnemar(a, b)
    assert (fixed, broken) == (3, 1)
    assert p == pytest.approx(0.625, abs=1e-4)
    # Our verdict is stricter than AUDIT.md's prose: with 4 discordant pairs the
    # best reachable p is 0.125, so this could never have cleared 0.05 whatever
    # it measured. Both readings agree on the only thing that matters.
    line = verdict(fixed, broken, p)
    assert "significant" in line and "— significant" not in line


def test_the_measured_training_noise_reads_as_no_result():
    """AUDIT.md measured retraining noise at about one case: dispatcher-v2
    25/26 against the same corpus at seed 99, 24/26, p = 1.0. This is the
    single most important assertion in the file — it is the reason raw totals
    are banned.
    """
    a = [True] * 25 + [False]        # 25/26
    b = [True] * 24 + [False, True]  # 24/26, disagreeing on two cases
    fixed, broken, p = mcnemar(a, b)
    assert (fixed, broken) == (1, 1)
    assert p == 1.0


def test_identical_runs_are_no_difference_not_a_win():
    fixed, broken, p = mcnemar([True, False, True], [True, False, True])
    assert (fixed, broken, p) == (0, 0, 1.0)
    assert "no difference" in verdict(fixed, broken, p)


def test_a_paired_test_refuses_unaligned_results():
    """Two runs over different case counts are not paired, and quietly zipping
    them to the shorter one would produce a confident number about nothing."""
    with pytest.raises(ValueError, match="equal-length"):
        mcnemar([True, False], [True, False, True])


def test_discordant_ignores_cases_both_runs_agree_on():
    a = [True, True, False, False, True]
    b = [True, False, True, False, True]
    assert discordant(a, b) == (1, 1)


# ---------------------------------------------------------------------------
# the leak guard — one copy, at exactly 0.82


def test_threshold_is_exactly_the_measured_value():
    """0.82 in three separate files of the earlier harness. It is a measurement, not a
    preference: at this value the check found 12 rows that were gate cases with
    the numbers changed."""
    assert LEAK_THRESHOLD == 0.82


def test_a_gate_case_with_the_numbers_changed_is_caught():
    """The exact failure the threshold was set for."""
    guard = ["work out 88 times 14 for me"]
    assert too_close("work out 91 times 17 for me", guard)


def test_an_identical_case_is_caught():
    guard = ["what percentage is 47 out of 260"]
    assert too_close("what percentage is 47 out of 260", guard)


def test_a_genuinely_different_subject_passes():
    guard = ["work out 88 times 14 for me"]
    assert not too_close("what time is 09:00 in Tokyo if I'm in Berlin", guard)


def test_the_threshold_is_a_strict_greater_than():
    """A ratio of exactly 0.82 passes. Ported verbatim: the earlier harness's three
    copies all used `> threshold`, and flipping it to `>=` would change which
    rows the published audits rejected."""
    assert not too_close("abcd", ["abcd"], threshold=1.0)


def test_find_leaks_names_the_pair_and_sorts_worst_first():
    """A boolean is not actionable; the offending pair is."""
    guard = ["work out 88 times 14 for me", "convert 200 ml to cups"]
    rows = ["work out 89 times 14 for me", "hello there", "convert 200 ml to cups"]
    hits = find_leaks(rows, guard)
    assert len(hits) == 2
    assert hits[0][2] >= hits[1][2]
    assert all(len(h) == 3 for h in hits)
    assert "hello there" not in [h[0] for h in hits]


# ---------------------------------------------------------------------------
# the corpus audit — NOTES.md's 28:7


def _rows(labelled: int, bare: int):
    return ([(f"table row {i}", "labelled", "table:labelled") for i in range(labelled)]
            + [(f"bare row {i}", "bare", "table:bare") for i in range(bare)])


def test_the_28_to_7_imbalance_is_rejected():
    """NOTES.md: p3 had 28 labelled table rows against 7 bare, and learned
    "a table means labels=" so firmly it invented labels for every bare table.
    This is the case the guard exists for and it must fail."""
    problems = label_ratio_problems(_rows(28, 7))
    assert problems, "28:7 must be rejected — it cost a training run"
    assert any("28:7" in p for p in problems)


def test_balancing_the_halves_28_to_28_passes():
    """NOTES.md: "Balancing the halves 28:28 fixed it (bare 10/28 -> 18/28)".
    The fix that worked must pass the guard that rejected the failure."""
    assert label_ratio_problems(_rows(28, 28)) == []


def test_the_ratio_boundary_is_where_it_was_measured():
    assert MAX_LABEL_RATIO == 1.6
    assert label_ratio_problems(_rows(16, 10)) == []     # exactly 1.6:1
    assert label_ratio_problems(_rows(17, 10)) != []     # 1.7:1


def test_one_label_may_not_swallow_a_multi_label_corpus():
    """The 45% rule came from a corpus with many roles, where one label taking
    half the rows means the model can score well by always guessing it."""
    rows = ([(f"q{i}", "NONE", "") for i in range(50)]
            + [(f"p{i}", "skill", "") for i in range(30)]
            + [(f"r{i}", "search", "") for i in range(20)])
    problems = label_ratio_problems(rows)
    assert any("50/100" in p for p in problems)


def test_two_balanced_labels_are_not_reported_as_dominance():
    """A contrast is two labels by construction, so 50/50 is the target shape.
    Firing the share limit on it would fire hardest on the ideal corpus."""
    rows = ([(f"q{i}", "labelled", "") for i in range(30)]
            + [(f"p{i}", "bare", "") for i in range(30)])
    assert label_ratio_problems(rows) == []


def test_an_empty_corpus_is_a_problem_not_a_pass():
    assert label_ratio_problems([]) == ["the corpus is empty"]


def test_contradictory_labels_are_caught():
    rows = [("draw a star", "icon-drawer", ""), ("draw a star", "svg-shapes", "")]
    problems = contradiction_problems(rows)
    assert len(problems) == 1
    assert "icon-drawer" in problems[0] and "svg-shapes" in problems[0]


def test_a_reused_phrasing_is_not_a_contradiction_when_keyed_on_the_image():
    """The second half of the vision identity bug. A vision corpus reuses one
    phrasing across every image on purpose; keyed on the question alone, the
    whole corpus reads as contradictory. Measured on the dog corpus: two false
    findings covering all 130 rows."""
    rows = [("what breed is this?", "PUG", ""),
            ("what breed is this?", "BEAGLE", ""),
            ("what breed is this?", "BOXER", "")]
    assert contradiction_problems(rows) != [], "question-keyed, this looks broken"
    keys = ["pug-000.jpg", "beagle-000.jpg", "boxer-000.jpg"]
    assert contradiction_problems(rows, keys) == []


def test_one_image_with_two_labels_is_still_caught():
    """Keying on the image must not disable the check — it must make it ask the
    question that matters. The same photo labelled two ways is a real fault."""
    rows = [("what breed is this?", "PUG", ""),
            ("is this a pug?", "BEAGLE", "")]
    problems = contradiction_problems(rows, keys=["pug-000.jpg", "pug-000.jpg"])
    assert len(problems) == 1
    assert "BEAGLE" in problems[0] and "PUG" in problems[0]


def test_keys_must_line_up_with_rows():
    """A short key list would silently mis-pair rows with identities, which is
    worse than refusing."""
    rows = [("a", "1", ""), ("b", "2", "")]
    with pytest.raises(ValueError, match="one key per row"):
        contradiction_problems(rows, keys=["only-one"])


def test_audit_rows_passes_keys_through(rows_and_keys=None):
    """The whole-audit entry point has to honour the identity too, or the fix
    only works when the check is called directly."""
    rows = [("what breed is this?", "PUG", ""),
            ("what breed is this?", "BEAGLE", "")]
    keys = ["pug-000.jpg", "beagle-000.jpg"]
    problems, _ = audit_rows(rows, keys=keys)
    assert not any("contradictory" in p for p in problems)


def test_the_same_input_with_the_same_label_twice_is_fine():
    rows = [("draw a star", "icon-drawer", ""), ("draw a star", "icon-drawer", "")]
    assert contradiction_problems(rows) == []


# ---------------------------------------------------------------------------
# the sequence-length guard — the 924-token table


def test_number_dense_text_uses_the_dense_ratio():
    """NOTES.md: number-dense rows tokenize at 1.84 chars/token, not the ~3.5
    the app assumes. `84.39` is four tokens and a table is mostly that."""
    dense = "84.39,112.50,203.75,91.20,145.60,178.33," * 10
    assert estimate_tokens(dense) == pytest.approx(len(dense) / CHARS_PER_TOKEN_DENSE, rel=1e-6)


def test_the_prose_ratio_would_have_hidden_the_problem():
    """The measured failure: the prose ratio put a 924-token row at 527 and hid
    it completely. The dense estimate must be materially larger."""
    dense = "84.39,112.50,203.75,91.20," * 80
    assert estimate_tokens(dense) > len(dense) / 3.5 * 1.5


def test_a_row_past_the_ceiling_is_reported_as_truncation_not_rejection():
    """The trainer CUTS an over-long row rather than refusing it, and the answer
    is the tail — so this row would teach the model to stop mid-answer."""
    long_table = "84.39,112.50,203.75,91.20," * 80
    problems = sequence_length_problems([(long_table, "data=1,2,3", "table:bare")], seq_limit=1024)
    assert len(problems) == 1
    assert "ANSWER is what gets cut" in problems[0]


def test_an_ordinary_short_row_passes():
    assert sequence_length_problems([("add 5 and 7", "SKILL: calculator | 5 + 7", "")], 1024) == []


def test_the_system_prompt_counts_against_the_ceiling():
    """A 1,400-character system prompt is ~400 tokens the answer no longer has.
    Leaving it out of the estimate is how a row that fits in isolation gets
    truncated in training."""
    row = [("x" * 2800, "y" * 350, "")]          # ~912 tokens of prose alone
    assert sequence_length_problems(row, 1024) == []
    # A 1,400-character system prompt is ~400 more tokens the answer no longer has.
    assert sequence_length_problems(row, 1024, system_prompt_chars=1400) != []


# ---------------------------------------------------------------------------
# the whole audit


def test_audit_runs_every_check_and_reports_the_balance():
    rows = _rows(28, 7) + [("leaked case", "labelled", "table:labelled")]
    problems, balance = audit_rows(rows, guard=["leaked case"], seq_limit=1024)
    assert any("28:7" in p or "outnumbers" in p for p in problems)
    assert any("NEAR A HELD-OUT CASE" in p for p in problems)
    assert balance == {"labelled": 29, "bare": 7}


def test_free_text_answers_are_not_audited_for_balance():
    """A behavior cartridge has one distinct answer per row. "The most common
    answer appears twice" is not a finding, and a warning that fires on every
    build is a warning nobody reads."""
    rows = [(f"ticket {i}", f"a distinct subject line number {i}", "") for i in range(147)]
    assert label_ratio_problems(rows) == []


def test_a_classifier_with_many_rows_per_label_is_still_audited():
    """The escape hatch must not swallow the case the check exists for: a
    handful of labels over hundreds of rows is nowhere near the free-text line."""
    rows = ([(f"q{i}", "labelled", "") for i in range(280)]
            + [(f"p{i}", "bare", "") for i in range(70)])
    assert label_ratio_problems(rows) != []


def test_a_clean_corpus_produces_no_problems():
    rows = ([(f"draw icon number {i}", "icon-drawer", "a") for i in range(20)]
            + [(f"convert {i} ml to cups", "unit-converter", "b") for i in range(18)])
    problems, balance = audit_rows(rows, guard=["something else entirely"], seq_limit=1024,
                                   job_words=["draw an icon", "convert units"])
    assert problems == [], problems
    assert balance == {"icon-drawer": 20, "unit-converter": 18}


def test_the_seq_guard_is_skipped_rather_than_guessed_when_no_limit_is_given():
    """A guessed ceiling produces confident nonsense; absence is honest."""
    long_row = [("84.39,112.50," * 200, "data=1", "")]
    problems, _ = audit_rows(long_row)
    assert not any("token ceiling" in p for p in problems)


# ---------------------------------------------------------------------------
# subject variety — the earlier harness's plotter failure, as a check


def test_a_monotone_seed_set_trips_the_subject_check():
    """The gate for Phase 5: a deliberately one-subject practice set is caught.
    Every row here is about invoices; a model trained on it learns that the
    job is invoices."""
    rows = [(f"please summarize this invoice from supplier number {i} for the finance team", "ok", "") for i in range(20)]
    problems = subject_variety_problems(rows)
    assert problems, "twenty rows about invoices must be reported"
    assert any("'invoice'" in p and "20/20" in p for p in problems)


def test_a_varied_set_passes():
    subjects = ["a storm drain map", "a film storyboard", "quarterly revenue", "a habit tracker",
                "an article workflow", "a wedding seating chart", "a bus timetable", "a recipe",
                "a hiring pipeline", "a garden layout", "server uptime", "a reading list"]
    rows = [(f"draw {s} as a diagram", "ok", "") for s in subjects]
    assert subject_variety_problems(rows, job_words=["draw a diagram"]) == []


def test_the_jobs_own_words_are_allowed_everywhere():
    """'ticket' in every row of a ticket-subject corpus is the job, not a
    leaked subject. The plan's description says which words those are."""
    rows = [(f"ticket: {s}", "ok", "") for s in
            ["printer jams", "login fails", "invoice wrong", "vpn drops", "screen flickers",
             "email bounces", "badge broken", "laptop slow", "wifi weak", "mouse dead"]]
    assert subject_variety_problems(rows) != []
    assert subject_variety_problems(rows, job_words=["a support ticket description"]) == []


def test_too_few_rows_is_no_pattern():
    rows = [("invoice one", "a", ""), ("invoice two", "b", ""), ("invoice three", "c", "")]
    assert subject_variety_problems(rows) == []


def test_audit_rows_runs_the_subject_check():
    who = ["the bakery", "a dentist", "the garage", "a florist", "the school", "a plumber",
           "the library", "a tailor", "the cafe", "a vet", "the gym", "a printer"]
    rows = [(f"file the invoice for {w}", "ok", "") for w in who]
    problems, _ = audit_rows(rows)
    assert any("one subject dominates" in p and "'invoice'" in p for p in problems)
    problems, _ = audit_rows(rows, job_words=["file an invoice"])
    assert not any("one subject dominates" in p for p in problems)
