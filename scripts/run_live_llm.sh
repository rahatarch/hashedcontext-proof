#!/usr/bin/env bash
# Live-LLM diff-chain test against any OpenAI-compatible endpoint.
#
#   HC_API_BASE=https://api.example.com/v1 HC_MODEL=model-name scripts/run_live_llm.sh
#
# The API key is read from a file (default ~/.config/hashedcontext/api_key),
# never from the command line, and is never written to any output.
# Sends about 120 requests (~2.3M input tokens). Check your provider's pricing first.
source "$(dirname "$0")/_common.sh"
: "${HC_API_BASE:?set HC_API_BASE, e.g. https://api.deepseek.com}"
: "${HC_MODEL:?set HC_MODEL, e.g. deepseek-chat}"
KEY_FILE="${HC_API_KEY_FILE:-$HOME/.config/hashedcontext/api_key}"
[ -s "$KEY_FILE" ] || { echo "API key file not found: $KEY_FILE (see docs/07-live-llm-tests.md)"; exit 1; }
export HC_API_KEY="$(tr -d '[:space:]' < "$KEY_FILE")"
mkdir -p results
CASES="${CASES:-30}"
step "Diff-chain test, all three layouts ($CASES cases)"
$PY -m llm_eval.chain_eval --cases "$CASES" --question anchor | tail -n 40
step "Mode B with max_diffs_before_rebase=5 ($CASES cases)"
$PY -m llm_eval.chain_eval --cases "$CASES" --question anchor --rebase 5 --conds lazy | tail -n 15
step "Compare with published-results/"
$PY scripts/compare_with_published.py || true
