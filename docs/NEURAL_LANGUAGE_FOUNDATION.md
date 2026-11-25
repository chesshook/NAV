# MCR-L 0.8 — learned language foundation

## Why the architecture changes

The 0.2–0.7 releases built an inspectable symbolic prototype: token patterns are mapped
to a small semantic graph, a finite planner selects a supported act, and a grammar
realizes a response. This is useful for deterministic capabilities and precise
regression tests, but it does not generalize to ordinary language. The recent failures
("are u alive?", "can you teach me?", and a context-dependent "really?") are examples
of the same architectural limit, not three unrelated missing patterns.

MCR-L 0.8 adds a pretrained language model as the natural-language substrate. The
symbolic parser is no longer asked to understand arbitrary conversation.

## The new pipeline

The intended processing order is:

1. **Contextual interpretation.** The local model receives the current utterance and
   a bounded transcript. It produces a typed semantic frame: communicative act,
   intent, topic (or null), clause roles, references to earlier turns, knowledge need,
   ambiguity, and response goal.
2. **Deliberation (optional).** The --think option performs a separate planning call.
   It chooses the response operation, records the evidence/context it will use, names
   unresolved uncertainty, and states response constraints. It does not emit private
   chain-of-thought.
3. **Realization.** A separate model call writes the natural-language answer from the
   semantic frame and plan. It must respond to the meaning and context, not just echo
   the last token.
4. **Deterministic capabilities.** The bounded arithmetic AST evaluator remains a
   separate route. The old symbolic engine remains available as a baseline, but
   neural mode does not silently fall back to its keyword parser if the model server
   fails.

This separation makes the semantic interpretation inspectable. It does not make a
model's interpretation automatically correct: the frame is a hypothesis produced by
a learned model and must be evaluated against held-out conversations.

## Local runtime

MCR-L uses an OpenAI-compatible local chat-completions endpoint. llama.cpp is the
initial runtime because it supports GGUF models and constrained JSON output. The
Python code uses only the standard library; the model weights and inference server are
separate local components.

### Windows setup

1. Install a current llama.cpp build:

       winget install llama.cpp

2. Start the model server in its own terminal:

       scripts\start-local-model.cmd

   First launch downloads the Q4_K_M GGUF quantization. The script requests CPU-only
   inference (-ngl 0), binds only to localhost, and limits the context to 8K tokens.
   Keep that terminal running.

3. In a second terminal, from the NAV repository root, start learned-language mode:

       py -3 -m mcrl --neural --interactive --debug

4. Enable the additional deliberation pass:

       py -3 -m mcrl --neural --think --interactive --debug

The standard mode makes two model calls per user turn (understand, then realize).
--think makes three (understand, plan, realize). The extra planning pass costs
additional CPU time. The --think option is a structured MCR-L planning stage; it is
not a claim that this small model reproduces DeepSeek-R1's training or reasoning
capability.

### Model configuration

Defaults:
- API: http://127.0.0.1:8080/v1
- model identifier: gguf/Qwen3.5-2B-GGUF:Q4_K_M
- API key: none
- per-call timeout: 180 seconds

Override them with --model-url, --model-name, --model-api-key, and --model-timeout,
or the MCRL_MODEL_URL, MCRL_MODEL_NAME, and MCRL_MODEL_API_KEY environment variables.

The selected Qwen3.5 model is a 2B-parameter research/development-scale model. The
original Qwen model card lists Apache-2.0. Check the terms of the exact GGUF
quantization distribution you download as well. A small quantized model is a
CPU-accessible starting point, not a substitute for a larger model's capacity.

## Semantic frame and response plan

The frame deliberately separates these fields:

- speech_act: what the utterance is doing in this turn;
- intent: the user's practical conversational goal;
- topic: an actual world subject, or null if no such subject is present;
- references: prior propositions/turns needed to understand the utterance;
- clauses: predicate, agent, patient, polarity, modality, and time;
- requires_knowledge and knowledge_target: whether a world-knowledge answer is
  needed, and what it is about;
- needs_clarification, ambiguity, and response_goal;
- constraints: negation, requested output, boundaries, and other requirements.

The planner then decides whether to answer from the dialogue context, use model
knowledge, use supplied evidence, ask for clarification, acknowledge, perform a safe
action, explain a capability limit, or decline.

The semantic frame is constrained to a JSON schema by the local inference server and
validated again by MCR-L. This controls output shape; it does not prove that the
model understood the utterance correctly.

## Knowledge and trust

The learned model contains parametric knowledge from its pretraining and post-training.
That is different from facts verified against a current external source. MCR-L should
not invent citations, claim that parametric knowledge was externally checked, or
confuse a confident tone with evidence. Later work must add explicit retrieval,
source dates, provenance, and conflict handling for facts that need current
verification.

## Evaluation gate — what must improve

Do not judge this milestone by whether it can answer a few hand-picked prompts. The
new evaluation set must include:
- unseen paraphrases and informal spelling;
- role reversals and negation-scope changes;
- short reactions that refer to earlier assistant statements;
- embedded/quoted instructions and conflicting constraints;
- requests to teach without a supplied subject;
- identity and capability questions;
- topic changes, pronouns, and multi-turn references;
- factual questions where the model should state uncertainty;
- long transcripts and history recaps without truncation.

Report semantic-frame accuracy, reference-resolution accuracy, clarification precision,
response appropriateness, hallucination rate, latency, and peak RAM separately. Split
by conversation, not by individual sentence, to avoid leaking near-identical turns
between train and test. Keep the held-out test set frozen.

## What is not done yet

This release does not fine-tune a model, create new pretrained weights, or provide
GPT/DeepSeek-level ability. It adds an inference and semantic-interface foundation.
The next research stage is to curate a licensed, diverse conversation corpus with
human-reviewed semantic frames, build a reproducible train/validation/test pipeline,
then compare prompting, lightweight adapters, and distilled student models on held-out
generalization. Training a useful broad language model from random initialization is
not realistic on an ordinary CPU-only PC.
