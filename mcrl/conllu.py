"""Minimal streaming CoNLL-U reader for Universal Dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ConlluToken:
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str


def iter_conllu(path: str | Path) -> Iterator[list[ConlluToken]]:
    """Yield basic syntactic tokens; skip MWT range rows and empty decimal nodes."""
    sentence: list[ConlluToken] = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            line = raw.rstrip("\r\n")
            if not line:
                if sentence:
                    yield sentence
                    sentence = []
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError(
                    f"Invalid CoNLL-U line {line_number}: expected 10 columns, got {len(fields)}"
                )
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue
            try:
                index = int(token_id)
                head = int(fields[6])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer ID/HEAD in CoNLL-U line {line_number}"
                ) from exc
            if index < 1 or head < 0:
                raise ValueError(f"Invalid token index or head in line {line_number}")
            sentence.append(ConlluToken(
                id=index,
                form=fields[1],
                lemma=fields[2],
                upos=fields[3],
                xpos=fields[4],
                feats=fields[5],
                head=head,
                deprel=fields[7],
                deps=fields[8],
                misc=fields[9],
            ))
        if sentence:
            yield sentence
