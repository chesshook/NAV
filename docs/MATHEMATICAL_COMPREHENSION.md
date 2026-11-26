# MCR-L 1.0 — independent mathematical comprehension core

## Why v1.0 is different

MCR-L 0.8/0.9 has a Qwen-backed dialogue mode. In that mode, Qwen still performs broad sentence and discourse interpretation. That is not the independent mathematical understanding system targeted by this milestone.

MCR-L 1.0 adds a separate mathematical mode. It does not call Qwen, another LLM, an embedding endpoint, or a remote service at inference time. It performs lexical activation against a persistent dictionary graph, local part-of-speech decisions using a trained sparse averaged perceptron, dependency-structure decisions using a trained arc-standard transition parser, compositional extraction of clauses and grammatical roles, and an inspectable meaning structure.

This mode is an understanding/analysis stage. It deliberately does not invent factual answers or pretend it has already learned a natural conversation policy. Its output is the structure it believes the utterance expresses. The later chat-teaching and response-generation stage is deferred until meaning analysis is measured.

## Runtime architecture

    Raw utterance
      -> token / morphology / informal-form normalization
      -> lexical graph (candidate senses, phrase concepts, typed relations, unknown units)
      -> sparse POS perceptron
      -> sparse dependency transition perceptron
      -> compositional meaning graph (clauses, subjects, objects, negation, modality, questions)

Every token, including punctuation, is retained in the syntactic sequence. Unknown lexical items still receive a learned syntactic analysis and remain explicit unresolved dictionary nodes. They are not silently discarded.

The dictionary supplies lexical candidates and word-sense relations. It does not decide the grammar or the speaker's intention. The dependency parser supplies syntactic relations. It does not automatically know a syntactic subject's thematic role or prove that a proposition is true. The meaning composer preserves those distinctions rather than inventing semantic certainty.

## The learned syntax model

The parser uses two sparse averaged-perceptron models:
- A greedy POS tagger predicts Universal POS tags from lexical, affix, shape, neighbour, previous-tag, and dictionary-POS features.
- An arc-standard transition parser predicts SHIFT, LEFT-ARC, and RIGHT-ARC actions with typed Universal Dependency labels.

This is a CPU model with sparse feature weights. It is not a transformer, a token generator, or a Qwen wrapper. The training corpus provides examples of how words and grammatical patterns combine. The dictionary supplies lexical concepts and candidate senses.

For feature vector x and label y, the perceptron uses a linear score:

    score(y | x) = sum(weight[feature, label] for each active feature)

A mistake updates the gold and predicted labels:

    weight[feature, gold] += 1
    weight[feature, predicted] -= 1

Weights are averaged over training steps before the model is saved. These scores are not calibrated probabilities.

The dependency parser currently uses projective arc-standard trees. Non-projective training sentences are skipped and counted, not silently forced into an invalid tree. The held-out evaluation reports POS accuracy, unlabeled attachment score (UAS), labeled attachment score (LAS), and complete-tree fraction.

## Train the model on Windows

From the repository root:

    scripts\train-mathematical-core.cmd

Equivalent command:

    py -3 tools\train_mathematical_core.py --download --epochs 5

The script downloads the official Universal Dependencies English Web Treebank CoNLL-U train, development, test, and license files into data\training\UD_English-EWT\. It trains on the train split, reports development metrics, then reports the test split once as a held-out measurement. It saves the learned parser to data\models\mcrl_syntax_v1.json.

Do not tune training settings against the held-out test split. Use development metrics for engineering choices; reserve a separate interview-style set for the actual generalization gate.

### Training data attribution and licensing

Source: https://github.com/UniversalDependencies/UD_English-EWT

Its treebank page reports 16,622 sentences and about 254,820 words, drawn from blogs, social/web text, reviews, email, and Yahoo! Answers. The UD English EWT annotations are licensed CC BY-SA 4.0. The underlying source texts may have separate rights; this project does not redistribute the downloaded treebank files or claim their original texts are public domain. The training script preserves the source link, license name, and corpus SHA-256 values in the model metadata.

References:
- https://universaldependencies.org/treebanks/en_ewt/index.html
- https://github.com/UniversalDependencies/UD_English-EWT
- https://universaldependencies.org/format.html

The first syntax corpus is English-focused. Other languages require their own licensed treebanks and evaluation.

## Import the lexical dictionary

    scripts\import-open-english-wordnet.cmd

This creates data\semantic_network.sqlite. Open English WordNet 2025+ supplies word forms, sense synsets, glosses, examples, and typed relations. Its source and attribution are documented in MATHEMATICAL_SEMANTIC_NETWORK.md.

## Run independent mathematical understanding

Once both local data files exist:

    py -3 -m mcrl --mathematical --interactive --debug --lexicon-db data\semantic_network.sqlite --syntax-model data\models\mcrl_syntax_v1.json

No llama.cpp server is started. No model download is needed at runtime. No Qwen request is made. The debug output includes each input token and its learned UPOS tag, each dependency head and typed relation, candidate WordNet senses and unresolved words, predicate/clause structure and grammatical arguments, explicit negation and auxiliary/modality evidence, ambiguity notes and parse completeness.

The short line beginning MCR-L meaning: is an automatically composed structural readout, not a conversational answer. The goal at this stage is to inspect whether the meaning graph matches what a human listener would infer.

One-shot example:

    py -3 -m mcrl --mathematical --debug --lexicon-db data\semantic_network.sqlite --syntax-model data\models\mcrl_syntax_v1.json "Can you teach me?"

## What this milestone still does not solve

This is a first independent compositional syntax/lexical model, not complete understanding of all possible language. The initial syntax learner is trained only on English EWT's domains; semantic roles such as agent, experiencer, instrument, and recipient are candidates inferred from syntax and case markers, not a fully grounded verb-frame model; idioms, sarcasm, metaphor, quotation scope, ellipsis, coreference, and long-range discourse can still be misinterpreted; WordNet does not encode all factual knowledge or the physical world; a syntax dependency tree is not proof that a speaker's claim is true; this version does not generate new conversational answers or learn behaviour from the chat itself.

The next milestone is not a response-template expansion. It is an independent sentence/discourse-semantic learner: train and evaluate predicate/argument frames, scope, reference resolution, conditionals, comparisons, event structure, and ambiguity using licensed corpora. Only once these representations generalize to unseen interview conversations should chat-based teaching modify concept and behaviour memory with explicit provenance.

## Test suite

    py -3 -m unittest discover -s tests -v

Tests cover CoNLL-U ingestion, POS/dependency training, model save/load, retention of unknown words, subject/object composition, polarity, and interrogative-vs-declarative structure. Passing software tests does not establish broad interview comprehension; held-out corpus metrics and real conversations remain essential.
