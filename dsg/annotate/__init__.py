"""A local labelling app for the audit sheets, synced to GitHub.

Each annotator runs it on their own laptop against their own files
(``audit/<sheet>/annotators/<me>.csv``); it never opens anyone else's labels.

    python -m dsg.annotate --me annotator_01 --initials CB
"""
