# MCR-L 1.0
## Mathematical Cognitive Resonance for Language

MCR-L 1.0 adds an independent mathematical-comprehension mode. Unlike the earlier v0.8/v0.9 neural dialogue mode, the mathematical mode does not call Qwen or any LLM. It processes each utterance through a dictionary-backed lexical network, a locally trained sparse POS tagger, a dependency parser, and a compositional semantic graph.

This is the first independent comprehension foundation, not a claim of perfect understanding or a finished conversational assistant.

## The modes are deliberately different

- Mathematical mode: lexical graph + CPU-trained mathematical syntax model + compositional meaning analysis. No LLM inference. It returns an inspectable interpretation and ambiguity report, not a canned question answer.
- Neural mode: previous v0.8 Qwen-backed language interpretation and response realization. This remains available for comparison, but it is not the mode that implements the independent mathematical foundation.
- Default mode: previous symbolic baseline.

## Build the local data

### 1. Import the lexical network

    scripts\import-open-english-wordnet.cmd

This downloads Open English WordNet 2025+ and builds data\semantic_network.sqlite.

### 2. Train the mathematical syntax core

    scripts\train-mathematical-core.cmd

Equivalent:

    py -3 tools\train_mathematical_core.py --download --epochs 5

The trainer downloads Universal Dependencies English EWT train/dev/test CoNLL-U files, trains a sparse averaged-perceptron POS tagger and arc-standard dependency parser, evaluates the development and held-out test splits, and writes data\models\mcrl_syntax_v1.json.

The UD English EWT annotations are CC BY-SA 4.0. The script records attribution, license metadata, and SHA-256 checksums. The corpus's source texts may have separate rights; see its upstream README and license: https://github.com/UniversalDependencies/UD_English-EWT

### 3. Start the independent comprehension interview

    py -3 -m mcrl --mathematical --interactive --debug --lexicon-db data\semantic_network.sqlite --syntax-model data\models\mcrl_syntax_v1.json

There is no llama.cpp server to start for this mode. It runs locally on CPU, and does not request answers from Qwen.

One-shot inspection:

    py -3 -m mcrl --mathematical --debug --lexicon-db data\semantic_network.sqlite --syntax-model data\models\mcrl_syntax_v1.json "I do not like the way the system handles unknown words."

## What the mathematical mode computes

1. Preserves the token sequence, including punctuation and informal contractions.
2. Retrieves candidate dictionary senses for known words and phrases.
3. Uses dictionary POS hints plus learned contextual features to predict each token's Universal POS category.
4. Uses a learned dependency-transition network to predict the syntactic head and relation of each token.
5. Composes predicates, subjects, objects, oblique relations, modifiers, embedded clauses, polarity, auxiliaries, modality, and question/directive/assertion structure.
6. Keeps unknown lexical terms, ambiguous senses, and incomplete parses visible.

The meaning readout is a structural paraphrase of the current parse, not an answer template. Use --debug to see the complete meaning graph.

The sparse perceptron uses score(label | features) = sum(feature_weight[feature, label]). Weights are learned from a licensed syntactic corpus, not hand-authored per question. Its confidence is not calibrated, and generalization must be measured on unseen text.

## Important limitations

The initial grammar training data is English Web Treebank, about 16.6k sentences and 254.8k words, across web/social/blog/review/email genres. It gives the project a learned general syntax baseline, not all the grammatical or semantic knowledge of English. It will still make mistakes, especially with rare constructions, informal spelling, metaphor, sarcasm, nested scope, reference resolution, and world knowledge.

This mode does not yet answer factual questions or teach itself new behaviours in chat. The immediate next stage is to improve and evaluate independent compositional meaning on unseen interview-style conversations. Chat-based teaching comes only after the meaning representation can reliably show what was said and what remains ambiguous.

Full design and equations: docs/MATHEMATICAL_COMPREHENSION.md

## Run tests

    py -3 -m unittest discover -s tests -v
