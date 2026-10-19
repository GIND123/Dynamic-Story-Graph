# Labelling guide: revision calculus audit

A program read 85 novels and excerpts one window at a time. It kept a list of
facts about the story, such as "Harriet is at Highbury". Each row is a moment
when a new fact clashed with one it already held on the same slot (same person,
same kind of fact). You decide what that clash really was.

**Label only your own file**, `annotators/annotator_NN.csv`. It has no run,
condition or other hidden columns, so you can't tell which system wrote the
text. Some rows are in both annotators' files; we use them to report agreement
(Cohen's kappa). So label every row on your own, and don't compare answers until
both files are finished. Don't open `sheet.csv`, `sheet_blind.csv` or
`annotators/assignment.json`. Don't edit any column except the label columns:
the check script will reject a file whose row text has changed.

When you're done, run `python -m dsg.eval.labels check --sheet audit/calculus`
(add `--partial` to check a file before it's complete) and fix anything it lists.

Your file also hides what the program decided and why.

## What each row shows

- `subject`, `predicate`: the slot, e.g. `Harriet`, `location`.
- `old_object`, `old_evidence`, `old_context`: the earlier fact and where it came from.
- `new_object`, `new_evidence`, `new_context`: the new fact.
- In the context columns, `>>> <<<` marks the sentence the evidence was found
  in. `old_match` / `new_match` say how it was found:
  `exact`, `fuzzy` (the program's quote was paraphrased, so check the sentence
  really says it), or `not_found` (no sentence shown; read `old_window` /
  `new_window`, the full passage the fact came from).
- The text stops at the end of the new passage. Judge as a reader who has
  read the book up to that point and no further. Don't use what you know of
  the rest of the book.

## The label: `label`

The key question: **did the story world change, or was the earlier belief
wrong?** (This is the update-versus-revision distinction of Katsuno & Mendelzon,
1991. An *update* records a change in the world. A *revision* corrects a mistaken
belief about a world that did not change.)

| label | when | examples |
|---|---|---|
| `SUPERSEDE` | Both facts were true, at different times. The world changed. | Harriet *was* at Highbury, then *went* to London. Father was alive, then died. Jane lived at Longbourn, then moved. |
| `REVISE` | The earlier fact was never true. The reader, or a character the reader trusted, had it wrong. | "In fact he had never left London." The supposed uncle turns out to be her father. A disguise is dropped. |
| `ELABORATE` | No clash. The new value says the same thing more precisely, or fills in a gap. | "a house" → "Satis House". "London" → "a lodging in Holborn". "someone" → "Mr. Darcy". |
| `EXTRACTION_ERROR` | One of the two facts isn't actually in the text: wrong person, wrong slot, invented, or not a fact (e.g. `location = wretched`). | |
| `UNSURE` | You can't tell from the text shown. Add a note. | |

**Label the clash the text shows, not what you guess the program did.** For
many rows the program decided "no clash, just more detail" and quietly
replaced the old value with the new one. If the two values can't both be true
at once, that was a real clash. Label it `SUPERSEDE` or `REVISE` as the text
warrants, never `ELABORATE` by default. Use `ELABORATE` only when the new value
really is the old one stated more precisely.

Judgements that come up often:
- **Finding out is not the same as being wrong.** "She learned that her father
  had died" reports a *change* (SUPERSEDE). "She learned that he had never been
  her father" reports a *correction* (REVISE). Look at what was learned, not at
  the word "learned".
- **Who held the earlier belief matters.** A character's false claim that the
  narration later contradicts is REVISE. A character's true report that is
  later out of date is SUPERSEDE.
- **Two compatible values are not a clash.** "London" and "London Bridge" are
  ELABORATE only if the second really is a more precise version of the first,
  i.e. the same place. If they are different places, it is a change
  (SUPERSEDE) or an error.
- **Permanent facts** (parentage, siblings, birthplace, eye colour) can't
  change, so a clash there is usually REVISE or EXTRACTION_ERROR. Check anyway.
- If either fact is an extraction error, label `EXTRACTION_ERROR`, even if the
  other looks fine.

## Which value is correct: `correct_value`

At this point in the book (the end of the new passage), which value is true
*now*? Answer `old`, `new`, `both` (only when the two values are compatible, as
with ELABORATE), `neither` or `unsure`. For a SUPERSEDE the current value is
usually `new`. For a REVISE it is whichever value the correction lands on. If the right value is neither as written but
you can name it, put it in `correct_value_text`. We'll reuse these answers as
gold, so be exact.

## Notes
- Write your initials in `labeller`. Use `notes` freely, and always for `UNSURE`.
- Judge the two facts as stated, not the program's wording of the slot.
- About 30 rows take an hour.
