// Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------------------------------------------------
def NODE_LABEL = params.NODE_LABEL.trim()
def DOCKER_IMAGE = params.DOCKER_IMAGE.trim()
def PACKAGE_URL = params.PACKAGE_URL.trim()
def WHL_URL = params.WHL_URL.trim()
def BRANCH = params.BRANCH.trim()
def KEEP_TESTAGENT = params.KEEP_TESTAGENT
// ---------------------------------------------------------------------------------------------------------------------
pipeline {
    agent { node { label "normal_worker" } }
    stages {
        stage('test') {
            steps {
                script{
                    currentBuild.description = "SDK = <a href=\"${PACKAGE_URL}\">${PACKAGE_URL.split("corex/")[1]}</a>"
                    def build_job = build job: '../T_unit_LMcache-iluvatar',
                    parameters: [
                        string(name: 'ONE_CLICK', value: env.BUILD_URL),
                        string(name: 'NODE_LABEL', value: "${NODE_LABEL}"),
                        string(name: 'DOCKER_IMAGE', value: "${DOCKER_IMAGE}"),
                        string(name: 'PACKAGE_URL', value: "${PACKAGE_URL}"),
                        string(name: 'WHL_URL', value: "${WHL_URL}"),
                        string(name: 'BRANCH', value: "${BRANCH}"),
                        booleanParam(name: 'KEEP_TESTAGENT', value: "${KEEP_TESTAGENT}"),
                    ], wait: true
                }
            }
        }
    }
}
