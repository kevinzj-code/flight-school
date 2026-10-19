"""pigeon/support.py against Pigeon's own highlighter tests (material.rs), on the
fixture they use, so both apps pick the same sentence for the same answer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pigeon import support as s  # noqa: E402

ARTICLE_II_1 = (Path(__file__).parent / "fixtures/article-ii-section-1.txt").read_text()
SENATE = ("The Senate of the United States shall be composed of two Senators from each State, chosen by the Legislature thereof, "
          "for six Years; and each Senator shall have one Vote.\n\nNo Person shall be a Senator who shall not have attained to the "
          "Age of thirty Years, and been nine Years a Citizen of the United States, and who shall not, when elected, be an "
          "Inhabitant of that State for which he shall be chosen.\n\nThe Vice President of the United States shall be President "
          "of the Senate, but shall have no Vote, unless they be equally divided.")


def marks(section, answer):
    got = s.support_scored(section, answer)
    assert got, f"nothing marked for {answer!r}"
    at, also, _ = got
    return section[at[0]:at[1]], (section[also[0]:also[1]] if also else None)


def test_the_supporting_sentence_is_the_one_that_shares_the_answers_words():
    at, _ = marks(SENATE, "At least thirty Years old. [§4]")
    assert at.startswith("No Person shall be a Senator who shall not have attained to the Age of thirty Years")
    assert at.endswith("for which he shall be chosen.")
    at, _ = marks(SENATE, "Six Years, with two Senators from each State, each having one Vote. [§4]")
    assert at.startswith("The Senate of the United States")
    assert s.support_scored(SENATE, "It repeals the eighteenth article. [§46]") is None
    assert s.support_scored(SENATE, "[§4]") is None


def test_a_number_is_matched_however_it_is_written():
    age = "neither shall any Person be eligible to that Office who shall not have attained to the Age of thirty five Years"
    for answer in ["The president must be at least thirty-five years old to be eligible for the office. [§12]",
                   "The Constitution requires the president to be at least thirty-five years old.",
                   "The president must be at least 35 years old.",
                   "A president must have been a resident for 14 years."]:
        at, also = marks(ARTICLE_II_1, answer)
        assert at.startswith(age), (answer, at)
        assert also is None
    at, _ = marks(ARTICLE_II_1, "The president serves a four-year term.")
    assert at.startswith("He shall hold his Office during the Term of four Years")
    assert marks("It weighs 3 kilos. It is 35 years old.", "It is thirty-five.")[0] == "It is 35 years old."


def test_tokens_ordinals_and_numbers():
    assert s.support_tokens("the twenty-first article") == ["xxi", "article"]
    assert s.support_tokens("thirty-five 35 07") == ["thirty", "five", "thirty", "five", "seven"]
    assert s.support_tokens("1808 and 3.8") == ["1808", "three", "eight"]


def test_a_tie_goes_to_the_rarer_words_and_two_clauses_mark_both():
    section = "The President shall hold Office. The President shall leave Office. The President shall be thirty."
    assert marks(section, "The President is in office at thirty.")[0] == "The President shall be thirty."
    at, also = marks(ARTICLE_II_1, "The president must be a natural born citizen and at least thirty-five years old.")
    assert at.startswith("No Person except a natural born Citizen")
    assert also and also.startswith("neither shall any Person be eligible")
    at, also = marks(ARTICLE_II_1, "The day electors vote is the same throughout the United States.")
    assert at == "which Day shall be the same throughout the United States." and also is None


def test_sentences_end_at_stops_and_paragraphs():
    text = "Café first; then second. Third?\n\nFourth"
    assert [text[a:b] for a, b in s.sentences(text)] == ["Café first;", "then second.", "Third?", "Fourth"]


EXPECTED = "No person except a natural born Citizen shall be eligible to the Office of President. [§2]"


def test_same_source_is_the_same_sentence_whatever_the_wording():
    m = s.same_source(2, ARTICLE_II_1, EXPECTED, "Only a natural born citizen can be eligible to become President. [§2]")
    assert m.same and m.expected == m.got and "natural born Citizen" in m.got
    m = s.same_source(2, ARTICLE_II_1, EXPECTED, "The President holds office for a term of four years. [§2]")
    assert not m.same and "four Years" in m.got


def test_citations_and_refusals_decide_a_source_too():
    same = lambda e, g: s.same_source(2, ARTICLE_II_1, e, g).same  # noqa: E731
    assert not same(EXPECTED, "Only a natural born citizen is eligible to be President.")
    assert not same(EXPECTED, "Only a natural born citizen is eligible to be President. [§2] [§9]")
    assert same("Only natural born citizens are eligible to be President.", "A natural born citizen is the only person eligible for President. [§2]")
    no = "The passage does not say how a President is removed."
    assert same(no, "The section does not mention removal. [§2]")
    m = s.same_source(2, ARTICLE_II_1, EXPECTED, no)
    assert not m.same and m.got is None
    assert not same(EXPECTED, "Bananas are yellow. [§2]")
