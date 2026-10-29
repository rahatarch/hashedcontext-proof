#!/usr/bin/env bash
# Full reproduction of every published number except the live-LLM test.
# Time: ~20 minutes on a 16-thread desktop, 1-2 hours on a 4-core laptop.
# Results go to results/, figures to figures/.
source "$(dirname "$0")/_common.sh"
mkdir -p results
step "1/10 Unit and property tests";                   $PY -m pytest -q tests
step "2/10 Fetch the four pinned repositories";        $PY -m sim.fetch_corpus
step "3/10 Agent fleet: 20,000 sessions";              $PY -m experiments.e1_fleet 20000 > results/e1_fleet.log && echo "  done (details: results/e1_fleet.log)"
step "4/10 GPU serving simulator: sweep + cluster";    $PY -m experiments.e2_serving all | tail -n 5
step "5/10 HTTP gateway load test (64 clients x 20)";  $PY -m gateway.loadtest 64 20
step "6/10 Micro-benchmarks";                          $PY -m experiments.e4_micro > /dev/null
step "7/10 Re-derive the paper's arithmetic";          $PY -m experiments.e5_paper_math 67 > /dev/null
step "8/10 Needle test (mechanical, 300 cases)";       $PY -m llm_eval.needle_eval --mechanical --cases 300 > /dev/null
step "9/10 Diff-chain harness (mechanical)";           $PY -m llm_eval.chain_eval --mechanical --cases 100 --question anchor > /dev/null
step "10/10 Summary, figures and trade-off tables";    $PY -m experiments.analyze > /dev/null && $PY -m experiments.tradeoff_tables > /dev/null
step "Compare with published-results/";                $PY scripts/compare_with_published.py
