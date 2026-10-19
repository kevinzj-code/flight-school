//! A Reader's material: plain text in, an ordered section index out.
//!
//! Everything here is deterministic code — chunking, token arithmetic and
//! citation checking sit on this side, and no model ever decides where a
//! section starts. Removed with the lessons in d0a0d14, back for the Reader
//! (the kinds design note, phase B); `material.rs` is the file layer over it.
//!
//! Shapes:
//! - Markdown splits on ATX headings (`#`..`######`), keeping each heading
//!   with its body.
//! - HTML is reduced to text first (tags stripped, scripts/styles dropped),
//!   with <h1>-<h6> boundaries surviving as markdown-style headings so the
//!   markdown splitter can do the rest.
//! - Plain text (and extracted PDF text) first has its heading-shaped lines
//!   promoted to markdown headings — "Article I", "Section 3", "AMENDMENT XIV",
//!   a line underlined with === — and splits on those; text with no such
//!   lines splits on blank-line paragraphs, merged up to ~600 estimated tokens
//!   per chunk.
//!
//! Token counts are ESTIMATES (chars/3.5, the same rule the earlier Airoplane Mode app uses
//! for its context gauge). They gate warnings and merging, not training — the
//! real tokenizer count happens once, in the sequence-length preflight.

use serde::{Deserialize, Serialize};

/// Estimated tokens for a piece of text. Deliberately the same ratio the
/// earlier Airoplane Mode app's estTokens uses, so the two apps' gauges agree about the
/// same material.
pub fn est_tokens(s: &str) -> u32 {
    (s.chars().count() as f64 / 3.5).ceil() as u32
}

/// One section of one reference file. `ordinal` is 1-based across the whole
/// lesson (file order, then section order) — it is the label the UI shows
/// ("§4"), the id questions carry in their `source_section`, and the unit the
/// coverage readout counts.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Chunk {
    pub ordinal: u32,
    pub file: String,
    /// Heading text when the source had one; a generated "Part N" label when
    /// it split on paragraphs.
    pub title: String,
    pub text: String,
    pub est_tokens: u32,
}

/// The whole index for a lesson's refs/ folder, persisted as chunks.json.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ChunkIndex {
    pub chunks: Vec<Chunk>,
    pub total_est_tokens: u32,
}

/// Merge target for structureless text. Small enough that one chunk plus a
/// question and an answer sits comfortably in a drafting call.
const MERGE_TARGET_TOKENS: u32 = 600;

// ---------------------------------------------------------------------------
// splitting

/// Markdown → (title, body) sections on ATX headings. Text before the first
/// heading becomes its own untitled section rather than being lost.
fn split_markdown(text: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut current_title = String::new();
    let mut current: Vec<&str> = Vec::new();
    let mut in_fence = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        // A `# heading` inside a code fence is code, not structure.
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
        }
        let is_heading = !in_fence
            && trimmed.starts_with('#')
            && trimmed.chars().take_while(|c| *c == '#').count() <= 6
            && trimmed
                .chars()
                .skip_while(|c| *c == '#')
                .next()
                .map(|c| c == ' ')
                .unwrap_or(false);
        if is_heading {
            if !current.is_empty() || !current_title.is_empty() {
                sections.push((current_title.clone(), current.join("\n")));
            }
            current_title = trimmed.trim_start_matches('#').trim().to_string();
            current = Vec::new();
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() || !current_title.is_empty() {
        sections.push((current_title, current.join("\n")));
    }
    sections
        .into_iter()
        .map(|(t, b)| (t, b.trim().to_string()))
        .filter(|(t, b)| !t.is_empty() || !b.is_empty())
        .collect()
}

/// The six entities the strip decodes, in one pass with everything else.
/// Anything beyond these is left as written — this is a tag stripper, not an
/// HTML parser, and it is honest about that.
const ENTITIES: [(&str, char); 6] = [
    ("&amp;", '&'),
    ("&lt;", '<'),
    ("&gt;", '>'),
    ("&quot;", '"'),
    ("&#39;", '\''),
    ("&nbsp;", ' '),
];

/// Tags whose CONTENTS go too, not just the tags.
const DROPPED_ELEMENTS: [&str; 2] = ["script", "style"];

/// Tags that become a line break, so paragraph structure survives the strip.
const BREAK_TAGS: [&str; 5] = ["p", "br", "div", "li", "tr"];

/// Where the element opened just before `from` ends — one past its closing
/// tag, or the end of the document when it never closes.
fn skip_element(html: &str, from: usize, name: &str) -> usize {
    let mut j = from;
    while let Some(rel) = html[j..].find('<') {
        let at = j + rel;
        if let Some(tail) = html[at + 1..].strip_prefix('/') {
            // Byte comparison, not a slice: `name` is ASCII by construction,
            // but `tail` may start mid-way through a multi-byte character and
            // slicing it by `name.len()` would panic.
            let n = name.len();
            if tail.len() >= n && tail.as_bytes()[..n].eq_ignore_ascii_case(name.as_bytes()) {
                return html[at..].find('>').map(|p| at + p + 1).unwrap_or(html.len());
            }
        }
        j = at + 1;
    }
    html.len()
}

/// HTML → text with heading boundaries preserved as `# `-style lines, then
/// through the markdown splitter. Not a real parser — a tag stripper that is
/// honest about what it is. Scripts and styles are dropped wholesale; every
/// other tag becomes a space; entities beyond the common six stay as-is.
///
/// One pass, no allocation per tag, and text runs are copied as string slices
/// rather than byte by byte. That last part is not a micro-optimisation: the
/// byte-at-a-time version pushed each byte `as char`, which reads UTF-8 as
/// Latin-1 and turned every non-ASCII character in a reference file into
/// mojibake — and that text goes on to become training data.
pub fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let mut i = 0;
    // Start of the run of literal text not yet copied into `out`.
    let mut text_from = 0;

    while i < bytes.len() {
        if bytes[i] == b'&' {
            if let Some((entity, ch)) = ENTITIES.iter().find(|(e, _)| html[i..].starts_with(*e)) {
                out.push_str(&html[text_from..i]);
                out.push(*ch);
                i += entity.len();
                text_from = i;
                continue;
            }
            i += 1;
            continue;
        }
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        out.push_str(&html[text_from..i]);

        // The tag body, between '<' and the next '>'. An unclosed tag at the
        // end of the document swallows the rest, the way a browser does.
        let (body, after) = match html[i + 1..].find('>') {
            Some(p) => (&html[i + 1..i + 1 + p], i + 1 + p + 1),
            None => (&html[i + 1..], html.len()),
        };
        let closing = body.starts_with('/');
        let named = body.trim_start_matches('/');
        let name_len = named.bytes().take_while(u8::is_ascii_alphanumeric).count();
        let name = &named[..name_len];

        if !closing && DROPPED_ELEMENTS.iter().any(|t| name.eq_ignore_ascii_case(t)) {
            i = skip_element(html, after, name);
            text_from = i;
            continue;
        }

        let heading_level = (name.len() == 2 && name.as_bytes()[0].eq_ignore_ascii_case(&b'h'))
            .then(|| (name.as_bytes()[1] as char).to_digit(10))
            .flatten()
            .filter(|l| (1..=6).contains(l));

        match heading_level {
            Some(level) if !closing => {
                out.push('\n');
                for _ in 0..level {
                    out.push('#');
                }
                out.push(' ');
            }
            // A closing </h2> ends the heading line.
            Some(_) => out.push('\n'),
            None if BREAK_TAGS.iter().any(|t| name.eq_ignore_ascii_case(t)) => out.push('\n'),
            None => out.push(' '),
        }
        i = after;
        text_from = i;
    }
    out.push_str(&html[text_from..]);
    out
}

/// Structureless text → paragraphs on blank lines, merged up to the target.
fn split_paragraphs(text: &str) -> Vec<String> {
    let mut merged: Vec<String> = Vec::new();
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        match merged.last_mut() {
            Some(last) if est_tokens(last) + est_tokens(para) <= MERGE_TARGET_TOKENS => {
                last.push_str("\n\n");
                last.push_str(para);
            }
            _ => merged.push(para.to_string()),
        }
    }
    merged
}

/// The kinds of file a Reader accepts. A PDF arrives here as text already —
/// `material.rs` reads it through PDFKit, and Vision's text recognition for
/// a page with no text layer — and is chunked like plain text.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RefKind {
    Markdown,
    Html,
    Plain,
    Pdf,
}

pub fn kind_for_ext(ext: &str) -> Option<RefKind> {
    match ext.to_lowercase().as_str() {
        "md" | "markdown" => Some(RefKind::Markdown),
        "html" | "htm" => Some(RefKind::Html),
        "txt" | "text" => Some(RefKind::Plain),
        "pdf" => Some(RefKind::Pdf),
        _ => None,
    }
}

/// Words that open a heading in the documents people actually bring: law,
/// manuals, handbooks. A line that starts with one of these, followed by a
/// number or a numeral, and is short, is a section title and not a sentence.
const HEADING_WORDS: [&str; 10] = ["article", "amendment", "section", "chapter", "part", "title", "appendix", "preamble", "schedule", "rule"];

fn is_roman_or_number(w: &str) -> bool {
    let w = w.trim_end_matches(|c: char| c == '.' || c == ':' || c == ')');
    !w.is_empty() && (w.chars().all(|c| c.is_ascii_digit()) || w.chars().all(|c| "ivxlcdmIVXLCDM".contains(c)))
}

/// Whether one line of structureless text is a heading. Deliberately narrow:
/// a false heading splits a section in two, which is survivable; a missed one
/// leaves the paragraph splitter to do its work, which is what happened
/// before this existed.
fn looks_like_heading(line: &str, next: Option<&str>) -> bool {
    let t = line.trim();
    if t.len() < 2 || t.len() > 80 || t.ends_with(',') || t.ends_with(';') {
        return false;
    }
    // Underlined the markdown way: the next line is === or ---.
    if let Some(n) = next {
        let n = n.trim();
        if n.len() >= 3 && (n.chars().all(|c| c == '=') || n.chars().all(|c| c == '-')) {
            return true;
        }
    }
    let mut words = t.split_whitespace();
    let first = words.next().unwrap_or("").to_lowercase();
    let first = first.trim_end_matches(|c: char| c == '.' || c == ':');
    if HEADING_WORDS.contains(&first) {
        return words.next().is_some_and(is_roman_or_number) && t.len() <= 60;
    }
    // ALL CAPS, at least two words, some letters: a title set the old way.
    let letters: Vec<char> = t.chars().filter(|c| c.is_alphabetic()).collect();
    letters.len() >= 4 && letters.iter().all(|c| c.is_uppercase()) && t.split_whitespace().count() >= 2
}

/// Structureless text with its heading-shaped lines promoted to `## `
/// headings, so the markdown splitter can section it. Returns None when no
/// line qualified, and the caller falls back to paragraphs.
pub fn promote_headings(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = String::with_capacity(text.len() + 64);
    let mut found = 0;
    let mut skip = false;
    for (i, line) in lines.iter().enumerate() {
        if skip { skip = false; continue; }
        let next = lines.get(i + 1).copied();
        if looks_like_heading(line, next) {
            let t = line.trim();
            // An underline belongs to the heading, not the body.
            if next.is_some_and(|n| { let n = n.trim(); n.len() >= 3 && (n.chars().all(|c| c == '=') || n.chars().all(|c| c == '-')) }) {
                skip = true;
            }
            out.push_str("\n## ");
            out.push_str(t);
            out.push('\n');
            found += 1;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    (found > 0).then_some(out)
}

/// Chunk one file's text. `start_ordinal` is where numbering continues from,
/// so ordinals are unique across the whole lesson.
pub fn chunk_file(file: &str, kind: RefKind, text: &str, start_ordinal: u32) -> Vec<Chunk> {
    let sections: Vec<(String, String)> = match kind {
        RefKind::Markdown => split_markdown(text),
        RefKind::Html => split_markdown(&html_to_text(text)),
        RefKind::Plain | RefKind::Pdf => match promote_headings(text) {
            Some(md) => split_markdown(&md),
            None => split_paragraphs(text)
                .into_iter()
                .enumerate()
                .map(|(i, p)| (format!("Part {}", i + 1), p))
                .collect(),
        },
    };
    sections
        .into_iter()
        .filter(|(_, body)| !body.trim().is_empty())
        .enumerate()
        .map(|(i, (title, body))| Chunk {
            ordinal: start_ordinal + i as u32,
            file: file.to_string(),
            title: if title.is_empty() { format!("Part {}", i + 1) } else { title },
            est_tokens: est_tokens(&body),
            text: body,
        })
        .collect()
}

// `haystacks`, `best_chunk_for_in` and `coverage` are the reference
// definition of `keyword-overlap-v1` (docs/PIGEON-FORMAT.md §8), which
// Pigeon's own material.rs mirrors at ask time. Flight School does not
// retrieve — the rows carry their section — so nothing here calls them; they
// stay, tested, so the two apps have one definition to agree on.
#[allow(dead_code)]
/// The lowercased `title + text` of every chunk, in chunk order — the
/// haystacks `best_chunk_for_in` searches.
///
/// Split out and cached by the caller on purpose: grounding runs once per
/// drafted answer, and building these inside the search meant allocating two
/// full copies of the entire reference corpus for every question asked.
pub fn haystacks(index: &ChunkIndex) -> Vec<String> {
    index
        .chunks
        .iter()
        .map(|c| {
            let mut h = String::with_capacity(c.title.len() + 1 + c.text.len());
            h.push_str(&c.title.to_lowercase());
            h.push(' ');
            h.push_str(&c.text.to_lowercase());
            h
        })
        .collect()
}

/// Deterministic best-chunk pick for a question with no section reference:
/// keyword overlap between the question's words and each chunk, longest
/// words weighted. Below the floor, the honest answer is "no chunk" — the
/// caller then tells the model the manual doesn't cover it rather than
/// grounding the answer in noise.
///
/// `hay` must be `haystacks(index)`, positionally aligned with `index.chunks`;
/// a shorter slice simply stops the search early rather than mis-scoring.
#[allow(dead_code)]
pub fn best_chunk_for_in<'a>(
    index: &'a ChunkIndex,
    hay: &[String],
    question: &str,
) -> Option<&'a Chunk> {
    let words: Vec<String> = question
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 4)
        .map(str::to_string)
        .collect();
    if words.is_empty() {
        return None;
    }
    let mut best: Option<(&Chunk, u32)> = None;
    for (chunk, hay) in index.chunks.iter().zip(hay) {
        let score: u32 = words
            .iter()
            .map(|w| if hay.contains(w.as_str()) { w.len() as u32 } else { 0 })
            .sum();
        if score > best.map(|(_, s)| s).unwrap_or(0) {
            best = Some((chunk, score));
        }
    }
    // The floor: at least two matched medium words' worth of signal.
    best.filter(|(_, s)| *s >= 9).map(|(c, _)| c)
}

// ---------------------------------------------------------------------------
// keyword-overlap-v2 — the reference definition. Pigeon's material.rs carries
// a byte-for-byte port; the two must agree, because the section this picks is
// the section the model answers from, and a different pick reads exactly like
// the right one.
//
// Measured on the Constitution Reader (the grounded-Pigeon brief): v1
// found the right section for 76 of 176 questions, because "President",
// "Congress" and "State" are everywhere and the longest section wins. v2 is
// BM25 (k1 1.2, b 0.75) over stemmed words of three letters or more with a
// stoplist, an ordinal word ("fifteenth") read as the numeral its heading
// carries ("xv"), a bonus for a word in the section's title, and a bonus for
// a two-word phrase the question and the section share. 117 of 176. The base
// model's own embeddings, tried the same day, found 8 — so this stays lexical.

// The v2 and hybrid-v1 references are called by their tests and by Pigeon's
// port; Flight School itself does not retrieve, so the compiler sees them
// unused. They stay, tested, so the two apps have one definition.
#[allow(dead_code)]
const STOP: [&str; 46] = ["the", "and", "for", "that", "with", "this", "shall", "from", "into", "upon", "than", "them", "they",
    "their", "there", "which", "what", "who", "whom", "when", "where", "how", "does", "did", "have", "has", "had", "been",
    "being", "are", "was", "were", "not", "nor", "but", "any", "all", "such", "may", "can", "each", "one", "two", "under",
    "over", "other"];
#[allow(dead_code)]
const STOP_TAIL: [&str; 4] = ["out", "its", "his", "her"];
#[allow(dead_code)]
const ORDINALS: [&str; 20] = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth"];
#[allow(dead_code)]
const V2_K1: f64 = 1.2;
#[allow(dead_code)]
const V2_B: f64 = 0.75;
#[allow(dead_code)]
const V2_TITLE_BONUS: f64 = 2.0;
#[allow(dead_code)]
const V2_PHRASE_BONUS: f64 = 2.0;
#[allow(dead_code)]
const V2_FLOOR: f64 = 3.0;

#[allow(dead_code)]
fn roman(n: u32) -> String {
    let mut n = n;
    let mut out = String::new();
    for (v, s) in [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")] {
        while n >= v { out.push_str(s); n -= v; }
    }
    out
}

#[allow(dead_code)]
fn ordinal_number(w: &str) -> Option<u32> {
    if let Some(i) = ORDINALS.iter().position(|o| *o == w) { return Some(i as u32 + 1); }
    let rest = w.strip_prefix("twenty")?;
    let rest = rest.strip_prefix('-').unwrap_or(rest);
    ORDINALS.iter().take(7).position(|o| *o == rest).map(|i| 21 + i as u32)
}

#[allow(dead_code)]
fn stem(w: &str) -> String {
    if w.len() > 4 && w.ends_with("ies") { return format!("{}y", &w[..w.len() - 3]); }
    if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") { return w[..w.len() - 1].to_string(); }
    w.to_string()
}

/// The words v2 scores: lowercased, split on anything but letters, digits and
/// hyphens; an ordinal word becomes the numeral a heading would carry;
/// hyphens are dropped; stopwords and words under three letters go.
#[allow(dead_code)]
pub fn v2_tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut out = Vec::new();
    for raw in lower.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-')) {
        if raw.is_empty() { continue; }
        let joined: String = raw.chars().filter(|c| *c != '-').collect();
        if let Some(n) = ordinal_number(raw).or_else(|| ordinal_number(&joined)) { out.push(roman(n)); continue; }
        if joined.len() >= 3 && !STOP.contains(&joined.as_str()) && !STOP_TAIL.contains(&joined.as_str()) {
            out.push(stem(&joined));
        }
    }
    out
}

#[allow(dead_code)]
struct V2Doc {
    tf: std::collections::HashMap<String, u32>,
    len: usize,
    title: std::collections::HashSet<String>,
    bigrams: std::collections::HashSet<(String, String)>,
}

#[allow(dead_code)]
fn v2_doc(chunk: &Chunk) -> V2Doc {
    let words = v2_tokens(&format!("{} {}", chunk.title, chunk.text));
    let mut tf = std::collections::HashMap::new();
    for w in &words { *tf.entry(w.clone()).or_insert(0) += 1; }
    let bigrams = words.windows(2).map(|p| (p[0].clone(), p[1].clone())).collect();
    V2Doc { tf, len: words.len(), title: v2_tokens(&chunk.title).into_iter().collect(), bigrams }
}

/// `keyword-overlap-v2`: the one best section for a question, or none when
/// the best score is under the floor. Ties go to the earlier section.
#[allow(dead_code)]
pub fn best_chunk_v2<'a>(index: &'a ChunkIndex, question: &str) -> Option<&'a Chunk> {
    let n = index.chunks.len();
    if n == 0 { return None; }
    let docs: Vec<V2Doc> = index.chunks.iter().map(v2_doc).collect();
    let avg = docs.iter().map(|d| d.len).sum::<usize>() as f64 / n as f64;
    let mut df: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for d in &docs { for w in d.tf.keys() { *df.entry(w.as_str()).or_insert(0) += 1; } }
    let idf = |w: &str| -> f64 { let f = *df.get(w).unwrap_or(&0) as f64; (1.0 + (n as f64 - f + 0.5) / (f + 0.5)).ln() };
    let q = v2_tokens(question);
    let mut qset: Vec<&str> = Vec::new();
    for w in &q { if !qset.contains(&w.as_str()) { qset.push(w); } }
    let mut qbi: Vec<(&str, &str)> = Vec::new();
    for p in q.windows(2) { let b = (p[0].as_str(), p[1].as_str()); if !qbi.contains(&b) { qbi.push(b); } }
    let mut best: Option<(usize, f64)> = None;
    for (i, d) in docs.iter().enumerate() {
        let mut s = 0.0;
        for w in &qset {
            let f = *d.tf.get(*w).unwrap_or(&0) as f64;
            if f > 0.0 {
                s += idf(w) * f * (V2_K1 + 1.0) / (f + V2_K1 * (1.0 - V2_B + V2_B * d.len as f64 / avg));
                if d.title.contains(*w) { s += V2_TITLE_BONUS * idf(w); }
            }
        }
        s += V2_PHRASE_BONUS * qbi.iter().filter(|(a, b)| d.bigrams.contains(&(a.to_string(), b.to_string()))).count() as f64;
        if best.map(|(_, b)| s > b).unwrap_or(true) { best = Some((i, s)); }
    }
    best.filter(|(_, s)| *s >= V2_FLOOR).map(|(i, _)| &index.chunks[i])
}

// ---------------------------------------------------------------------------
// hybrid-v1 — the reference definition of the blend (pigeon/embed.py measures
// with it, Pigeon's material.rs ports it). The embedding itself is not done
// here: Flight School embeds at pack time through llama-server (embed.py), and
// the reader embeds only the question. This is the arithmetic that turns a
// question's vector and the packed paragraph vectors into one section.
//
// Measured 2026-09-30 on the Constitution Reader, right section first out of
// 176: keyword-overlap-v2 117, paragraph embeddings alone 126, this blend 142
// with the packed 16-bit vectors (top three 164). One definition of the
// weights and the floor, on both sides.

#[allow(dead_code)]
pub const HYBRID_COSINE_WEIGHT: f64 = 0.7;
#[allow(dead_code)]
pub const HYBRID_LEXICAL_WEIGHT: f64 = 0.3;
/// The best paragraph's cosine under which the honest answer is no section:
/// the right section's cosine was 0.537 at least and 0.658 at the fifth
/// percentile; off-topic questions reached 0.49–0.61.
#[allow(dead_code)]
pub const HYBRID_COSINE_FLOOR: f64 = 0.58;

/// A packed unit vector: 16-bit little-endian floats, base64, as embed.py
/// writes them. Decoded by hand — no half-float crate on either side.
#[allow(dead_code)]
pub fn unpack_f16(b64: &str, dims: usize) -> Option<Vec<f32>> {
    let bytes = base64_decode(b64)?;
    if bytes.len() != dims * 2 { return None; }
    Some(bytes.chunks_exact(2).map(|p| half_to_f32(u16::from_le_bytes([p[0], p[1]]))).collect())
}

#[allow(dead_code)]
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 { sign << 31 } else {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 { f <<= 1; e -= 1; }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

#[allow(dead_code)]
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits = 0;
    for c in s.bytes() {
        if c == b'=' { break; }
        let v = T.iter().position(|t| *t == c)? as u32;
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 { bits -= 8; out.push(((buf >> bits) & 0xff) as u8); }
    }
    Some(out)
}

#[allow(dead_code)]
pub fn zscores(xs: &[f64]) -> Vec<f64> {
    let n = xs.len() as f64;
    if n == 0.0 { return Vec::new(); }
    let m = xs.iter().sum::<f64>() / n;
    let sd = (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt();
    xs.iter().map(|x| if sd > 0.0 { (x - m) / sd } else { 0.0 }).collect()
}

/// The section to paste: `cos` is each section's best paragraph cosine with
/// the question (-1 where it has none), `lexical` its keyword-overlap-v2
/// score, both in section order. None when the best section's cosine is
/// under the floor. Ties go to the earlier section.
#[allow(dead_code)]
pub fn hybrid_v1(cos: &[f64], lexical: &[f64]) -> Option<usize> {
    if cos.is_empty() || cos.len() != lexical.len() { return None; }
    let (zc, zl) = (zscores(cos), zscores(lexical));
    let mut best: Option<(usize, f64)> = None;
    for i in 0..cos.len() {
        let s = HYBRID_COSINE_WEIGHT * zc[i] + HYBRID_LEXICAL_WEIGHT * zl[i];
        if best.map(|(_, b)| s > b).unwrap_or(true) { best = Some((i, s)); }
    }
    best.filter(|(i, _)| cos[*i] >= HYBRID_COSINE_FLOOR).map(|(i, _)| i)
}

/// Every section's best paragraph cosine with a unit-length question vector,
/// from packed units of (section index, vector).
#[allow(dead_code)]
pub fn best_cosines(n: usize, units: &[(usize, Vec<f32>)], q: &[f32]) -> Vec<f64> {
    let mut cos = vec![-1.0f64; n];
    for (i, v) in units {
        let d: f64 = v.iter().zip(q).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        if d > cos[*i] { cos[*i] = d; }
    }
    cos
}

/// The lexical half of hybrid-v1: keyword-overlap-v2's raw score per section.
#[allow(dead_code)]
pub fn v2_scores(index: &ChunkIndex, question: &str) -> Vec<f64> {
    let n = index.chunks.len();
    if n == 0 { return Vec::new(); }
    let docs: Vec<V2Doc> = index.chunks.iter().map(v2_doc).collect();
    let avg = docs.iter().map(|d| d.len).sum::<usize>() as f64 / n as f64;
    let mut df: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for d in &docs { for w in d.tf.keys() { *df.entry(w.as_str()).or_insert(0) += 1; } }
    let idf = |w: &str| -> f64 { let f = *df.get(w).unwrap_or(&0) as f64; (1.0 + (n as f64 - f + 0.5) / (f + 0.5)).ln() };
    let q = v2_tokens(question);
    let mut qset: Vec<&str> = Vec::new();
    for w in &q { if !qset.contains(&w.as_str()) { qset.push(w); } }
    let mut qbi: Vec<(&str, &str)> = Vec::new();
    for p in q.windows(2) { let b = (p[0].as_str(), p[1].as_str()); if !qbi.contains(&b) { qbi.push(b); } }
    docs.iter().map(|d| {
        let mut s = 0.0;
        for w in &qset {
            let f = *d.tf.get(*w).unwrap_or(&0) as f64;
            if f > 0.0 {
                s += idf(w) * f * (V2_K1 + 1.0) / (f + V2_K1 * (1.0 - V2_B + V2_B * d.len as f64 / avg));
                if d.title.contains(*w) { s += V2_TITLE_BONUS * idf(w); }
            }
        }
        s + V2_PHRASE_BONUS * qbi.iter().filter(|(a, b)| d.bigrams.contains(&(a.to_string(), b.to_string()))).count() as f64
    }).collect()
}

/// Coverage: which section ordinals have at least one live question pointing
/// at them. Returns (covered, total).
#[allow(dead_code)]
pub fn coverage(index: &ChunkIndex, referenced: &[u32]) -> (u32, u32) {
    let total = index.chunks.len() as u32;
    let seen: std::collections::HashSet<u32> = referenced.iter().copied().collect();
    let covered = index
        .chunks
        .iter()
        .filter(|c| seen.contains(&c.ordinal))
        .count() as u32;
    (covered, total)
}

// ---------------------------------------------------------------------------
// the material block, and checkable citations

/// The exact text that fills the `{material}` slot of a grounded Pigeon's
/// prompt template.
///
/// ONE definition on purpose. The training rows are written with this and the
/// question is asked with this, and a grounded adapter is only correct in the
/// shape it was trained in — a passage pasted one way at training time and
/// another at ask time degrades quietly instead of failing, which is the whole
/// reason `prompt_template` travels inside the file at all.
///
/// The `[§n]` label is what makes a citation checkable: the ordinal is the
/// section's own id, so an answer that writes `[§4]` is making a claim the app
/// can verify against what it actually showed the model.
///
/// This and the two citation readers below are the reference definitions
/// `src/state.js` mirrors (`materialBlock`, `citedOrdinals`,
/// `unsupportedCitations`); the page builds rows itself, so nothing here calls
/// them outside the tests.
#[allow(dead_code)]
pub fn material_block(chunk: &Chunk) -> String {
    format!("[§{}] {}\n{}", chunk.ordinal, chunk.title, chunk.text)
}

/// The section ordinals an answer claims, in the `[§n]` form `material_block`
/// writes. Deduplicated, in order of first appearance.
#[allow(dead_code)]
pub fn cited_ordinals(answer: &str) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let bytes = answer.as_bytes();
    let mut i = 0usize;
    while let Some(pos) = answer[i..].find("[§") {
        let start = i + pos + "[§".len();
        let mut end = start;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end > start && bytes.get(end) == Some(&b']') {
            if let Ok(n) = answer[start..end].parse::<u32>() {
                if !out.contains(&n) {
                    out.push(n);
                }
            }
        }
        i = start;
    }
    out
}

/// Citations an answer makes that were never shown to it — the deterministic
/// half of grounding, needing no judge.
///
/// Measured on `runs/manual-answerer-v1` (the baseline notes, "Step 0 of the document
/// Pigeon"): handed nothing relevant, that adapter invented an answer and cited
/// a source anyway in 11 of 14 cases, and the base model did the same in 10 of
/// 14 — so an unverified citation is the normal behaviour of an untrained
/// grounded model, not an edge case. A citation nobody checks is decoration.
#[allow(dead_code)]
pub fn unsupported_citations(answer: &str, shown: &[u32]) -> Vec<u32> {
    cited_ordinals(answer)
        .into_iter()
        .filter(|n| !shown.contains(n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(ordinal: u32, title: &str, text: &str) -> Chunk {
        Chunk { ordinal, file: "manual.md".into(), title: title.into(),
                text: text.into(), est_tokens: est_tokens(text) }
    }

    #[test]
    fn a_material_block_labels_the_section_with_its_own_ordinal() {
        let b = material_block(&chunk(4, "Memory", "It needs about 3.8 GB."));
        assert_eq!(b, "[§4] Memory\nIt needs about 3.8 GB.");
    }

    #[test]
    fn citations_are_read_back_out_of_an_answer_deduplicated_and_in_order() {
        let a = "It needs 3.8 GB [§4], and the cap is separate [§7]. See also [§4].";
        assert_eq!(cited_ordinals(a), vec![4, 7]);
    }

    /// A number that is not a citation must not become one: the whole point of
    /// the marker is that "4 GB" and "[§4]" are different claims.
    #[test]
    fn plain_numbers_and_malformed_markers_are_not_citations() {
        assert!(cited_ordinals("4 GB, about 75%, section 4, [4], [§], [§x]").is_empty());
    }

    #[test]
    fn a_citation_of_a_section_that_was_never_shown_is_reported() {
        // Shown §4; the answer also claims §9, which it was never given.
        assert_eq!(unsupported_citations("true [§4] and also [§9]", &[4]), vec![9]);
        assert!(unsupported_citations("true [§4]", &[4, 7]).is_empty());
        // The measured failure: cites with nothing shown at all.
        assert_eq!(unsupported_citations("the manual says so [§1]", &[]), vec![1]);
    }

    #[test]
    fn markdown_splits_on_headings_and_keeps_preamble() {
        let md = "intro before any heading\n\n# One\nbody one\n\n## Two\nbody two";
        let s = split_markdown(md);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].0, "");
        assert!(s[0].1.contains("intro"));
        assert_eq!(s[1].0, "One");
        assert_eq!(s[2].0, "Two");
    }

    #[test]
    fn heading_inside_code_fence_is_code() {
        let md = "# Real\n```\n# not a heading\n```\nafter";
        let s = split_markdown(md);
        assert_eq!(s.len(), 1, "the fenced # must not split the section");
        assert!(s[0].1.contains("# not a heading"));
    }

    #[test]
    fn html_headings_survive_the_strip() {
        let html = "<html><head><style>h1{color:red}</style><script>var x=1;</script></head>\
                    <body><h1>Title</h1><p>First para</p><h2>Sub</h2><p>Second</p></body></html>";
        let text = html_to_text(html);
        assert!(!text.contains("color:red"), "style contents must be dropped");
        assert!(!text.contains("var x"), "script contents must be dropped");
        let s = split_markdown(&text);
        let titles: Vec<&str> = s.iter().map(|(t, _)| t.as_str()).collect();
        assert!(titles.contains(&"Title"));
        assert!(titles.contains(&"Sub"));
    }

    #[test]
    fn paragraphs_merge_to_target_not_past_it() {
        let para = "word ".repeat(700); // ~1000 est tokens — never merges
        let small = "tiny paragraph";
        let text = format!("{small}\n\n{small}\n\n{para}");
        let merged = split_paragraphs(&text);
        assert_eq!(merged.len(), 2, "two smalls merge; the big one stands alone");
        assert!(merged[0].contains(small));
    }

    /// The Constitution as plain text: articles and sections are lines of
    /// their own, and they must become the sections, not 600-token parts.
    #[test]
    fn plain_text_splits_on_article_and_section_lines() {
        let text = "Article I\n\nSection 1\nAll legislative Powers herein granted shall be vested in a Congress.\n\nSection 2\nThe House of Representatives shall be composed of Members.\n\nAMENDMENT XIV\nAll persons born or naturalized in the United States are citizens.";
        let chunks = chunk_file("c.txt", RefKind::Plain, text, 1);
        let titles: Vec<&str> = chunks.iter().map(|c| c.title.as_str()).collect();
        assert_eq!(titles, vec!["Section 1", "Section 2", "AMENDMENT XIV"], "got {titles:?}");
        assert!(chunks[0].text.contains("legislative"));
        assert!(chunks[2].text.contains("naturalized"));
    }

    #[test]
    fn a_sentence_that_starts_with_section_is_not_a_heading() {
        assert!(!looks_like_heading("Section 3 of the act says that every person who applies shall be heard in full, without exception.", None));
        assert!(!looks_like_heading("Part of the reason is cost,", None));
        assert!(looks_like_heading("Article IV.", None));
        assert!(looks_like_heading("Chapter 12: Memory", None));
        assert!(looks_like_heading("Underlined title", Some("=====")));
        assert!(!looks_like_heading("not a title", Some("more prose")));
    }

    #[test]
    fn text_with_no_headings_still_falls_back_to_paragraphs() {
        let text = "one paragraph of prose.\n\nanother paragraph of prose.";
        let chunks = chunk_file("p.txt", RefKind::Plain, text, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].title, "Part 1");
    }

    #[test]
    fn ordinals_continue_across_files() {
        let a = chunk_file("a.md", RefKind::Markdown, "# One\nx\n# Two\ny", 1);
        let b = chunk_file("b.txt", RefKind::Plain, "para", a.last().unwrap().ordinal + 1);
        assert_eq!(a[0].ordinal, 1);
        assert_eq!(a[1].ordinal, 2);
        assert_eq!(b[0].ordinal, 3);
    }

    /// The bug this replaced: the stripper pushed each BYTE `as char`, reading
    /// UTF-8 as Latin-1. Every non-ASCII character in a reference file became
    /// mojibake — and that text goes on to become training data.
    #[test]
    fn non_ascii_survives_the_strip() {
        let text = html_to_text("<p>caf\u{e9} \u{2014} na\u{ef}ve r\u{e9}sum\u{e9} \u{2713}</p>");
        assert!(text.contains("caf\u{e9}"), "got {text:?}");
        assert!(text.contains("na\u{ef}ve"), "got {text:?}");
        assert!(text.contains("\u{2014}"), "got {text:?}");
        assert!(text.contains("\u{2713}"), "got {text:?}");
    }

    /// The other bug: byte offsets from the ORIGINAL string were used to slice
    /// a LOWERCASED copy, which can be a different length. This input made the
    /// two desynchronise; it must strip, not panic.
    #[test]
    fn lowercasing_that_changes_length_does_not_panic() {
        let text = html_to_text("<H1>\u{130}STANBUL</H1><p>\u{130}\u{130}\u{130} body</p>");
        assert!(text.contains("STANBUL"), "got {text:?}");
        assert!(text.contains("# "), "the heading marker must survive: {text:?}");
    }

    #[test]
    fn entities_decode_in_the_same_pass() {
        assert_eq!(html_to_text("<p>a &amp; b &lt;c&gt; &quot;d&quot; &#39;e&#39;</p>").trim(),
                   "a & b <c> \"d\" 'e'");
        // &amp;lt; is a literal "&lt;", not a second round of decoding.
        assert_eq!(html_to_text("&amp;lt;").trim(), "&lt;");
    }

    /// The haystacks are built once and reused across many questions now, so
    /// the search has to stay case-insensitive and stable across calls — a
    /// stale or mis-cased haystack would ground answers in the wrong section.
    #[test]
    fn cached_haystacks_search_case_insensitively_and_repeatably() {
        let index = ChunkIndex {
            chunks: chunk_file(
                "m.md",
                RefKind::Markdown,
                "# Downloads\nModels download with resumable range requests.\n\
                 # Voice\nThe microphone uses Whisper for transcription.\n\
                 # Adapters\nAn adapter is refused against the wrong base model.",
                1,
            ),
            total_est_tokens: 0,
        };
        let hay = haystacks(&index);
        for (question, expected) in [
            ("how do model downloads resume?", Some(1)),
            ("WHISPER TRANSCRIPTION", Some(2)),
            ("refused against the wrong base", Some(3)),
            ("what colour is the sky?", None),
        ] {
            // Twice: reusing the haystacks must not change the answer.
            for _ in 0..2 {
                let got = best_chunk_for_in(&index, &hay, question).map(|c| c.ordinal);
                assert_eq!(got, expected, "on {question:?}");
            }
        }
    }

    #[test]
    fn best_chunk_needs_real_overlap() {
        let index = ChunkIndex {
            chunks: chunk_file(
                "m.md",
                RefKind::Markdown,
                "# Downloads\nModels download with resumable range requests.\n\
                 # Voice\nThe microphone uses whisper for transcription.",
                1,
            ),
            total_est_tokens: 0,
        };
        let hay = haystacks(&index);
        let hit = best_chunk_for_in(&index, &hay, "how do model downloads resume?").unwrap();
        assert_eq!(hit.title, "Downloads");
        assert!(
            best_chunk_for_in(&index, &hay, "what colour is the sky?").is_none(),
            "no-overlap questions get no chunk, not the least-bad one"
        );
    }

    /// v2 on the Constitution itself, against the dataset's own questions,
    /// so the reference and the number in the brief cannot drift apart
    /// unnoticed. Measured 2026-09-30 in Python with the same rules: 117 of
    /// 176. v1 found 76.
    #[test]
    fn v2_finds_the_right_section_for_two_thirds_of_the_constitution_questions() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../datasets/constitution-reader-v1/");
        let (Ok(md), Ok(qs)) = (std::fs::read_to_string(format!("{root}constitution.md")), std::fs::read_to_string(format!("{root}questions.json"))) else { return };
        let index = ChunkIndex { chunks: chunk_file("constitution.md", RefKind::Markdown, &md, 1), total_est_tokens: 0 };
        assert_eq!(index.chunks.len(), 52);
        let qs: Vec<serde_json::Value> = serde_json::from_str(&qs).unwrap();
        let mut right = 0;
        let mut none = 0;
        for q in &qs {
            match best_chunk_v2(&index, q["question"].as_str().unwrap()) {
                Some(c) if c.ordinal == q["section"].as_u64().unwrap() as u32 => right += 1,
                Some(_) => {}
                None => none += 1,
            }
        }
        eprintln!("v2: right {right} of {} · no section {none}", qs.len());
        assert!(right >= 115 && right <= 119, "right {right}: the port drifted from the measured 117");
        assert!(best_chunk_v2(&index, "how do I bake sourdough bread?").is_none(), "an off-topic question gets no section");
    }

    #[test]
    fn half_floats_and_base64_decode_by_hand() {
        // 0.5 = 0x3800, 1.0 = 0x3c00, -2.0 = 0xc000, in little-endian pairs.
        assert_eq!(unpack_f16("ADgAPADA", 3).unwrap(), vec![0.5, 1.0, -2.0]);
        assert!(unpack_f16("ADgAPADA", 4).is_none(), "a wrong length is refused");
        assert_eq!(half_to_f32(0x0001), 5.9604645e-8, "the smallest subnormal");
    }

    #[test]
    fn the_blend_weighs_cosine_over_words_and_has_a_floor() {
        // Two sections: the first wins on words, the second on meaning.
        assert_eq!(hybrid_v1(&[0.60, 0.80], &[9.0, 1.0]), Some(1));
        assert_eq!(hybrid_v1(&[0.80, 0.60], &[1.0, 9.0]), Some(0));
        assert_eq!(hybrid_v1(&[0.50, 0.55], &[9.0, 1.0]), None, "under the floor, no section");
        assert_eq!(hybrid_v1(&[0.70, 0.70], &[3.0, 3.0]), Some(0), "a tie goes to the earlier section");
        assert_eq!(hybrid_v1(&[], &[]), None);
    }

    /// The whole thing on the Constitution Reader's packed vectors, for the
    /// questions whose vectors ride in the dataset: the same picks Python made.
    #[test]
    fn hybrid_v1_reproduces_the_measured_picks_on_the_constitution() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../datasets/constitution-reader-v1/");
        let (Ok(md), Ok(emb), Ok(qs)) = (
            std::fs::read_to_string(format!("{root}constitution.md")),
            std::fs::read_to_string(format!("{root}embeddings.json")),
            std::fs::read_to_string(format!("{root}queries.json")),
        ) else { return };
        let index = ChunkIndex { chunks: chunk_file("constitution.md", RefKind::Markdown, &md, 1), total_est_tokens: 0 };
        let emb: serde_json::Value = serde_json::from_str(&emb).unwrap();
        let dims = emb["model"]["dims"].as_u64().unwrap() as usize;
        let pos = |o: u64| index.chunks.iter().position(|c| c.ordinal as u64 == o).unwrap();
        let units: Vec<(usize, Vec<f32>)> = emb["units"].as_array().unwrap().iter()
            .map(|u| (pos(u["ordinal"].as_u64().unwrap()), unpack_f16(u["f16"].as_str().unwrap(), dims).unwrap())).collect();
        let qs: Vec<serde_json::Value> = serde_json::from_str(&qs).unwrap();
        for q in &qs {
            let qv = unpack_f16(q["f16"].as_str().unwrap(), dims).unwrap();
            let cos = best_cosines(index.chunks.len(), &units, &qv);
            let lex = v2_scores(&index, q["question"].as_str().unwrap());
            let got = hybrid_v1(&cos, &lex).map(|i| index.chunks[i].ordinal as u64);
            assert_eq!(got, q["picked"].as_u64(), "{}", q["question"]);
        }
    }

    #[test]
    fn v2_tokens_read_ordinals_as_numerals_and_drop_stopwords() {
        assert_eq!(v2_tokens("What does the Fifteenth Amendment protect?"), vec!["xv", "amendment", "protect"], "{:?}", v2_tokens("What does the Fifteenth Amendment protect?"));
        assert_eq!(v2_tokens("the Twenty-second Amendment"), vec!["xxii", "amendment"]);
        assert_eq!(v2_tokens("Senators' duties and Treaties"), vec!["senator", "duty", "treaty"]);
    }

    #[test]
    fn coverage_counts_ordinals_once() {
        let index = ChunkIndex {
            chunks: chunk_file("m.md", RefKind::Markdown, "# A\nx\n# B\ny\n# C\nz", 1),
            total_est_tokens: 0,
        };
        assert_eq!(coverage(&index, &[1, 3, 3]), (2, 3));
        assert_eq!(coverage(&index, &[]), (0, 3));
    }
}
