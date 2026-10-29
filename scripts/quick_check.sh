#!/usr/bin/env bash
# Quick check (~5 minutes on a laptop): runs every deterministic part at a small
# scale and compares it with the published run. Results go to results/.
source "$(dirname "$0")/_common.sh"
mkdir -p results
step "1/7 Unit and property tests";            $PY -m pytest -q tests
step "2/7 Fetch the four pinned repositories"; $PY -m sim.fetch_corpus
step "3/7 Agent fleet: first 200 sessions";    $PY -m experiments.e1_fleet 200 > results/e1_fleet.log && echo "  done (details: results/e1_fleet.log)"
step "4/7 Re-derive the paper's arithmetic";   $PY -m experiments.e5_paper_math 67 > /dev/null
step "5/7 Micro-benchmarks";                   $PY -m experiments.e4_micro > /dev/null
step "6/7 Needle test (mechanical, 30 cases)"; $PY -m llm_eval.needle_eval --mechanical --cases 30 > /dev/null
step "7/7 Diff-chain harness (mechanical)";    $PY -m llm_eval.chain_eval --mechanical --cases 30 --question anchor > /dev/null
step "Compare with published-results/";        $PY scripts/compare_with_published.py
