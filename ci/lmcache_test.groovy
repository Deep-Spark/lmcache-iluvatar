// Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

@Library('jenkins_pipeline_shared_lib') _

runPipelineFramework(
    stage_list: [
        "lmcache_test": """
            set -euo pipefail
            bash scripts/install_dependencies.sh
            REPO_ROOT="\$(pwd)"
            tmpdir="\$(mktemp -d)"
            cd "\$tmpdir"
            python3 -m pytest -q --import-mode=append "\$REPO_ROOT/tests"
            WEIGHTS_ROOT="\${REPO_ROOT}/weights"
            MODEL_PATH="\${WEIGHTS_ROOT}/Qwen3-8B"
            mkdir -p "\${WEIGHTS_ROOT}"
            if [[ ! -f "\${MODEL_PATH}/config.json" ]]; then
                wget -q -O "\${WEIGHTS_ROOT}/Qwen3-8B.tar.gz" \\
                    http://sw.iluvatar.ai/download/apps/pretrained/nlp/Qwen3/Qwen3-8B.tar.gz
                tar -xzf "\${WEIGHTS_ROOT}/Qwen3-8B.tar.gz" -C "\${WEIGHTS_ROOT}"
            fi
            cd "\$REPO_ROOT/examples"
            python3 run_all_lmcache_iluvatar_tests.py \
                --base-model-path "\$MODEL_PATH" \
                --gpus 0,1,2,3 \
                --log-dir ../runtime_result \
                --keep-going
        """
    ]
)
