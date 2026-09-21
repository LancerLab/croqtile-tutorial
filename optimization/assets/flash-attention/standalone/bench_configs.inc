// Auto-generated from causal_prefill_d128/config.py -- do not edit
#pragma once

static const mha_helper::BenchConfig kBenchConfigs[] = {
    {4, 32, 4096, 4096, true, "B=4 H=32 SEQ=4096 causal prefill"},
    {4, 32, 8192, 8192, true, "B=4 H=32 SEQ=8192 causal prefill"},
    {4, 32, 16384, 16384, true, "B=4 H=32 SEQ=16384 causal prefill"},
};
static constexpr int kBenchConfigCount = sizeof(kBenchConfigs) / sizeof(kBenchConfigs[0]);
