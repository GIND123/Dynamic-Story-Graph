# Labelling guide: canon-violation detector audit

You are checking whether an automatic detector read a generated story correctly.
Each story was given planted facts in chapter 1, for example "Miriam has green
eyes" or "The locket is silver". The detector flags a chapter when a different
value seems to be given to the same subject. Your labels tell us how often it is
right.

**Label only your own file**, `annotators/annotator_NN.csv`. It has no run,
condition or other hidden columns, so you can't tell which system wrote the
text. Some rows are in both annotators' files; we use them to report agreement
(Cohen's kappa). So label every row on your own, and don't compare answers until
both files are finished. Don't open `sheet.csv`, `sheet_blind.csv` or
`annotators/assignment.json`. Don't edit any column except the label columns:
the check script will reject a file whose row text has changed.

When you're done, run `python -m dsg.eval.labels check --sheet audit/detector`
(add `--partial` to check a file before it's complete) and fix anything it lists.

Fill in only the empty label cells. Leave cells that say `n/a` alone.

## What each row shows

- `subject`, `kind`, `true_value`: the planted fact, e.g. `Miriam`, `eyes`, `green`.
- `context`: the passage. `[[ ]]` marks the subject mention the detector used.
  `<< >>` marks the word it matched.
- `item_type` tells you which question to answer:

| item_type | answer in | question |
|---|---|---|
| `flagged_violation` | `real_violation` | Does the `<<word>>` give the subject's *kind* a value other than `true_value`? |
| `flagged_restatement` | `real_restatement` | Does the `<<word>>` state the subject's *kind* as `true_value`? |
| `unflagged_mention` | both | Does anything in the passage give the subject's *kind* a wrong value (`real_violation`)? Does anything state it as `true_value` (`real_restatement`)? |

Answer `yes`, `no` or `unsure`. Write your initials in `labeller`. Use `notes`
for anything odd, and always add a note when you answer `unsure`.

## Deciding

Judge only the attribute named in `kind`, for the subject named in `subject`.
Use the text alone, not what the story "should" say.

**yes**: a reader would take the passage to say the subject has that value.
- The value is tied to the attribute: "Miriam's blue eyes", "Her blue eyes
  narrowed" (where *her* is Miriam), "Tobias ran a hand through his black hair".
- For metal: the value describes the object: "the gold locket", "The locket,
  gold and heavy", "The locket was gold".
- For trade: "Tobias, the apothecary", "Tobias worked as a blacksmith",
  "the apothecary Tobias".
- For home: "born in Ravensmoor", "Rowena, who came from Thornbury".
- Any value other than `true_value` counts, even one the planted list never
  offered: "brown eyes" contradicts green eyes, "the copper locket"
  contradicts a silver one, "Tobias the watchmaker" contradicts clockmaker
  only if a watchmaker is clearly a different trade to you (otherwise
  `unsure`, with a note).
- Dialogue counts when the story presents it as true. A lie that the story
  later exposes still counts as `yes` for that passage. Say so in `notes`.

**no**: the word is there but doesn't describe this attribute of this subject.
- It describes something else: "the grey morning", "the blue water",
  "his black coat", "the red door", "the gold light", "an iron nail".
- It belongs to another character: "Silas, the apothecary, nodded to Tobias".
  This is common: stories often plant the same kind of fact on two characters.
- It is part of a longer word for something else: "silver-white hair" is not
  a silver locket.
- It is negated or corrected nearby: "not gold but silver".
- It is a different attribute: "black hair" when `kind` is `eyes`.

**unsure**: you cannot tell who a pronoun refers to, the passage is cut off
before you can tell, or the text is too garbled to read. Don't guess.

## Notes
- One judgement per row. Don't let a neighbouring row influence it.
- If the passage contradicts the fact somewhere *other than* the marked word,
  answer for the marked word only, and mention the other place in `notes`.
- About 50 rows take an hour. Take breaks.
- Questions about a row: note the `item_id` and ask rather than skip.
