# MCR-L
## Mathematical Cognitive Resonance for Language

MCR-L 0.8 is an experimental CPU-first conversational AI architecture. It now has
two explicit modes:

- **Local learned-language mode:** uses a pretrained local model to interpret natural
  language in context, optionally plan a response, and realize it as normal prose.
- **Symbolic baseline:** keeps the earlier inspectable parser, deterministic dialogue
  constructions, source-backed fact store, and safe arithmetic evaluator.

The symbolic baseline is deliberately not presented as general language understanding.
A fixed parser cannot become GPT-like by accumulating more phrase-specific rules.

## The v0.8 language pipeline

In neural mode, the current utterance and a bounded transcript go through three
separate stages:

1. **Understand:** produce a structured semantic frame containing the user's
   communicative act, intent, actual topic (or null), clause roles, references to
   previous turns, knowledge need, ambiguity, and response goal.
2. **Think (optional):** use a separate structured deliberation pass to choose a
   response operation, identify context/evidence, note uncertainty, and set response
   requirements.
3. **Speak:** produce a natural-language answer from the semantic frame and plan.

The default neural path uses two model calls (understand, speak). Adding --think uses
three (understand, plan, speak). The plan is an inspectable decision summary, not
exposed chain-of-thought and not a claim to reproduce DeepSeek-R1 training.

The model output is constrained to a JSON schema and then validated by MCR-L. A valid
JSON frame does not prove semantic correctness; actual model evaluation is required.
When neural mode fails, MCR-L reports the failure instead of silently switching to
the symbolic keyword parser.

## Run locally on Windows

The initial model choice is Qwen3.5-2B in Q4_K_M GGUF form, served locally by llama.cpp.
The original Qwen3.5-2B model card lists Apache-2.0. Check the exact terms of the
quantized distribution you download. This 2B model is a practical CPU-first starting
point, not a promise of GPT-level breadth.

1. Install llama.cpp:

       winget install llama.cpp

2. Start its local model server from the repository root:

       scripts\start-local-model.cmd

   The first startup downloads the model. The script binds to localhost, sets an
   8K context, and requests CPU-only inference with -ngl 0. Keep this window open.

3. In a second terminal, start conversational mode:

       py -3 -m mcrl --neural --interactive --debug

4. To add the separate response-planning pass:

       py -3 -m mcrl --neural --think --interactive --debug

To run one prompt:

       py -3 -m mcrl --neural "Can you teach me?"

The defaults are API URL http://127.0.0.1:8080/v1 and model identifier
gguf/Qwen3.5-2B-GGUF:Q4_K_M. Override them using --model-url, --model-name,
--model-api-key, or --model-timeout. The corresponding environment variables are
MCRL_MODEL_URL, MCRL_MODEL_NAME, and MCRL_MODEL_API_KEY.

See docs/NEURAL_LANGUAGE_FOUNDATION.md for the full architecture, data contract,
runtime notes, evaluation gates, and limitations.

## Symbolic baseline

To run the previous deterministic engine without a language model:

       py -3 -m mcrl --interactive --debug

This mode is useful for examining the hand-built semantic graph, response planner,
and evidence rules. It has intentionally bounded language coverage and should not
be confused with the new learned-language mode.

Safe arithmetic remains deterministic and separate from the language model. It
parses a constrained numeric AST and does not use eval or exec.

## Evaluation

Run the unit tests:

       py -3 -m unittest discover -s tests -v

After starting the local model, run the initial multi-turn sanity set:

       py -3 tools\evaluate_neural_language.py

To include the extra planning pass:

       py -3 tools\evaluate_neural_language.py --think

The evaluation cases cover contextual reactions, teaching requests, assistant
identity, conversation history, instruction scope, corrections, and follow-up
references. This is an initial sanity set, not a statistically sufficient benchmark.
The next gate is a larger licensed and human-reviewed evaluation corpus with
conversation-level train/validation/test splits.

## What v0.8 does not claim

- It does not train a foundation model from scratch or fine-tune Qwen weights.
- It does not guarantee that the pretrained model's interpretation or answer is true.
- Parametric model knowledge is not the same as a fact verified against a current
  external source.
- The optional planning pass does not reproduce DeepSeek-R1's training, compute, or
  reasoning performance.
- CPU performance, peak RAM, semantic accuracy, hallucination rate, and language
  coverage still need to be measured on the target PC.

The purpose of v0.8 is to replace keyword-driven language interpretation with a real
learned language substrate and a typed interface that can be tested. Future milestones
must add retrieval with provenance, held-out evaluation, licensed semantic training
data, and carefully measured adapter/distillation experiments.
