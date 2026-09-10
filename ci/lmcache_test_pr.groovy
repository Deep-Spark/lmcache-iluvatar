// Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

@Library('jenkins_pipeline_shared_lib')_

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

def get_sdk_version_from_pr_title() {
    def pr_title = env.PR_TITLE ?: ""
    def matcher = (pr_title =~ ~/\[SDK-([\w.]+)\]/)
    return matcher.find() ? matcher.group(1) : null
}

def is_skip_ci() {
    return env.PR_TITLE?.contains('[SKIP_CI]') ?: false
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

def source_branch = env.SOURCE_BRANCH ?: params.SOURCE_BRANCH ?: 'master'
def target_branch = env.TARGET_BRANCH ?: params.TARGET_BRANCH ?: 'master'
def node_label = params.NODE_LABEL ?: 'mr-x86-8gpu || bi150-x86-4gpu'
def sdk_version = get_sdk_version_from_pr_title() ?: params.SDK_VERSION ?: '4.5.0.rc.8'
def packages = params.PACKAGES ?: 'corex-installer,cuda_python,torch,vllm,ixformer,cupy,triton,xformers,pycuda'

// Directory created after cloning the PR repo (matches the .git basename).
def repo_dir = 'lmcache-iluvatar'

def repo_list = [
    'ssh://git@bitbucket.iluvatar.ai:7999/swapp/lmcache-iluvatar.git': source_branch
]

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

def check_format = {
    sh """
        set -euo pipefail
        cd ${repo_dir}
        if [ ! -f .pre-commit-config.yaml ]; then
            echo "No .pre-commit-config.yaml found, skip format check"
            exit 0
        fi
        python3 -m pip install 'pre-commit>=3.5.0' --quiet
        git fetch origin ${target_branch} --depth=50
        mapfile -t CHANGED_FILES < <(git diff --name-only --diff-filter=ACMR origin/${target_branch}...HEAD)
        if [ \${#CHANGED_FILES[@]} -eq 0 ]; then
            echo "No changed files to check"
            exit 0
        fi
        printf 'Changed files (%d):\\n' "\${#CHANGED_FILES[@]}"
        printf '  %s\\n' "\${CHANGED_FILES[@]}"
        python3 -m pre_commit run --config .pre-commit-config.yaml --files "\${CHANGED_FILES[@]}"
    """
}

def build_lmcache = {
    sh """
        set -euo pipefail
        cd ${repo_dir}
        bash scripts/clean_build_install.sh
    """
}

def unit_test = {
    sh """
        set -euo pipefail
        REPO_ROOT="\$(cd ${repo_dir} && pwd)"
        tmpdir="\$(mktemp -d)"
        cd "\$tmpdir"
        python3 -m pytest -q --import-mode=append "\$REPO_ROOT/tests"
    """
}

def integration_test = {
    sh """
        set -euo pipefail
        REPO_ROOT="\$(cd ${repo_dir} && pwd)"
        WEIGHTS_ROOT="\${REPO_ROOT}/weights"
        MODEL_PATH="\${WEIGHTS_ROOT}/Qwen3-8B"
        mkdir -p "\${WEIGHTS_ROOT}"
        if [[ ! -f "\${MODEL_PATH}/config.json" ]]; then
            wget -q -O "\${WEIGHTS_ROOT}/Qwen3-8B.tar.gz" \\
                http://sw.iluvatar.ai/download/apps/pretrained/nlp/Qwen3/Qwen3-8B.tar.gz
            tar -xzf "\${WEIGHTS_ROOT}/Qwen3-8B.tar.gz" -C "\${WEIGHTS_ROOT}"
        fi
        cd "\${REPO_ROOT}/examples"
        python3 run_all_lmcache_iluvatar_tests.py \
            --base-model-path "\${MODEL_PATH}" \
            --gpus 0,1,2,3 \
            --log-dir ../runtime_result \
            --keep-going
    """
}

def stage_list = [
    check_format    : check_format,
    build_lmcache   : build_lmcache,
    unit_test       : unit_test,
    integration_test: integration_test,
]

// ---------------------------------------------------------------------------
// Pipeline entry point
// ---------------------------------------------------------------------------

// Skip CI
if (is_skip_ci()) {
    repo_list = []
    stage_list = []
    node_label = 'normal_worker'
}

runPipeline(
    package_url: "${sdk_version}:[${packages}]",
    repo_list: repo_list,
    stage_list: stage_list,
    node_label: node_label,
    timeout: 6
)
