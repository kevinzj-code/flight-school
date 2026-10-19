"""The fact swap finds numbers the way a document writes them.

Run: `python3 -m pytest src-tauri/python/bench/test_grounding.py -q`. No model.
The rows are in a Reader's shape — `[§n] Title`, the section, a blank line,
the question — with Constitution text, which writes nearly every number in
words: the probe that read digits only found no case in its exam at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instructor import grounding as g  # noqa: E402

SEC_9 = ("[§10] Article I, Section 9\n"
         "The Migration or Importation of such Persons as any of the States now existing shall "
         "think proper to admit, shall not be prohibited by the Congress prior to the Year one "
         "thousand eight hundred and eight, but a Tax or duty may be imposed on such Importation, "
         "not exceeding ten dollars for each Person.\n\n"
         "Until what year could Congress not prohibit the importation of persons?")
SEC_13 = ("[§13] Article II, Section 2\n"
          "He shall have Power, by and with the Advice and Consent of the Senate, to make Treaties, "
          "provided two thirds of the Senators present concur.\n\n"
          "How many Senators must concur in a treaty?")


def test_a_number_in_words_is_a_case():
    num = g.probe_number(SEC_9, "Until the year 1808. [§10]")
    assert num == "one thousand eight hundred and eight", "a digit answer matches the passage's words"
    altered, swap = g.swap_in(SEC_9, num)
    assert "Year one thousand seven hundred and seventy four" in altered
    assert "eight hundred and eight" not in altered


def test_a_fraction_swaps_whole():
    num = g.probe_number(SEC_13, "Two thirds of the Senators present. [§13]")
    assert num == "two thirds"
    altered, swap = g.swap_in(SEC_13, num)
    assert swap == "four ninths" and "provided four ninths of the Senators" in altered


def test_swapping_a_cardinal_leaves_a_fraction_alone():
    q = ("[§1] Article I, Section 3\nEach Senator shall have one Vote; two Senators from each State, "
         "and two thirds of the Members present.\n\nHow many Senators does each State have?")
    num = g.probe_number(q, "Two Senators. [§1]")
    altered, _ = g.swap_in(q, num)
    assert "two thirds" in altered, "the fraction is a different number and must survive"
    assert "Thirty nine Senators" in altered or "thirty nine Senators" in altered


def test_the_heading_number_is_never_the_fact():
    q = "[§11] Article I, Section 10\nNo State shall enter into any Treaty.\n\nMay a State make a treaty?"
    assert g.probe_number(q, "No — Article I, Section 10 forbids it. [§11]") is None
    altered, _ = g.swap_in("[§11] Article I, Section 10\nIt took 10 days.\n\nq", "10")
    assert altered.startswith("[§11] Article I, Section 10\n"), "the heading is left as written"


def test_one_is_not_a_number_worth_probing():
    q = "[§2] Article I, Section 2\nNo one shall be a Representative who is not a Citizen.\n\nq"
    assert g.probe_number(q, "No one who is not a Citizen. [§2]") is None


def test_verdict_reads_digits_and_words():
    num, swap = "one thousand eight hundred and eight", "one thousand seven hundred and seventy four"
    assert g.verdict("Until 1774.", num, swap) == "followed"
    assert g.verdict("Until the year one thousand eight hundred and eight.", num, swap) == "memorised"
    assert g.verdict("1808, or 1774 in this text", num, swap) == "both"
    assert g.verdict("Congress could not.", num, swap) == "neither"
    assert g.verdict("Twenty two of them", "two", "thirty nine") == "neither", \
        "'two' inside 'twenty two' is not a mention of two"


def test_digit_cases_are_unchanged():
    q = "[§4] Report\nRevenue rose 3.8 percent to 412 units.\n\nHow much did revenue rise?"
    assert g.probe_number(q, "3.8 percent. [§4]") == "3.8"
    altered, swap = g.swap_in(q, "3.8")
    assert swap == "8.2" and "rose 8.2 percent" in altered


def test_a_digit_swap_changes_only_that_number():
    q = "[§4] Report\nIt sold 12 units in 1812, then 4,012 in all, 12, as planned.\n\nq"
    assert g.probe_number(q, "12 units. [§4]") == "12"
    altered, swap = g.swap_in(q, "12")
    assert altered.endswith("It sold 49 units in 1812, then 4,012 in all, 49, as planned.\n\nq")


def test_to_words_round_trips():
    for n in (2, 13, 35, 40, 62, 100, 108, 1774, 1808, 2000, 25000):
        assert g.words_value(g.to_words(n)) == n, n


class EchoServer:
    """Answers with the passage's figure: reads it when the adapter is on, and
    'remembers' the original when it is off — a stand-in for both behaviours."""

    class spec:
        n_parallel = 1

    def chat(self, messages, *, lora=None, **kw):
        text = messages[-1]["content"]
        if "nothing to say" in text:
            return "It does not say."
        if lora is None:
            return "It is " + text.split("figure is ")[1].split(".")[0] + "."
        return "It is 12."


def test_a_case_counts_only_where_the_plain_answer_states_the_number():
    cases = [{"question": "[§1] T\nThe figure is 12.\n\nq", "altered": "[§1] T\nThe figure is 49.\n\nq",
              "num": "12", "swap": "49"},
             {"question": "[§2] T\nnothing to say. The figure is 12.\n\nq",
              "altered": "[§2] T\nnothing to say. The figure is 49.\n\nq", "num": "12", "swap": "49"}]
    got = g.run_swaps(EchoServer(), cases, max_tokens=8)
    assert got["adapter_n"] == 1 and got["adapter"]["followed"] == 1 and got["adapter"]["neither"] == 0, \
        "the case whose plain answer never states the number must not count as 'neither'"
    assert got["control_n"] == 1 and got["control"]["memorised"] == 1
    assert got["n"] == 2 and len(got["cases"]) == 2, "every case is still kept for reading"


def test_a_readers_rows_are_checked_on_their_source_sentence():
    """Kevin's test (2026-10-03): a right answer in other words is the same
    source; an answer from another sentence, or none, is not."""
    from instructor.fly import Flight, FlightRow
    chunks = [{"ordinal": 13, "title": "Article II, Section 2",
               "text": "He shall have Power, by and with the Advice and Consent of the Senate, to make Treaties, "
                       "provided two thirds of the Senators present concur. He shall nominate Ambassadors."}]
    expected = "Two thirds of the Senators present must concur. [§13]"
    row = lambda actual: FlightRow(question=SEC_13.replace("\n\nHow many", " He shall nominate Ambassadors.\n\nHow many"),  # noqa: E731
                                   expected=expected, actual=actual, group="rows", passed=False, how="judge")
    flight = Flight(rows=[row("A treaty needs two thirds of the Senators present to agree. [§13]"),
                          row("He nominates ambassadors. [§13]")],
                    control_rows=[row("I am not sure."), row("Two thirds of Senators present concur. [§13]")])
    got = g.source_check(flight, chunks)
    assert got == {"same": 1, "sentence": 1, "n": 2, "base_same": 1, "base_sentence": 1, "base_n": 2}
    # The untrained model's right sentence without a citation counts as the same sentence, not the same source.
    nocite = Flight(rows=[row("Two thirds of the Senators present.")])
    assert g.source_check(nocite, chunks) | {} == {"same": 0, "sentence": 1, "n": 1, "base_same": None, "base_sentence": None, "base_n": None}
    assert flight.rows[0].same_source["same"] and not flight.rows[1].same_source["same"]
    assert "Ambassadors" in flight.rows[1].same_source["got"]
    # Not a Reader's row (no section of its material): no score at all.
    assert g.source_check(Flight(rows=[row("x")]), [{"ordinal": 1, "title": "Other", "text": "Else."}]) is None
    assert g.source_check(flight, None) is None
