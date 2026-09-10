# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

mooncake_master \
  --rpc_port=15051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_port=18080 \
  --enable_offload=true \
  --root_fs_dir=/data/tmp/.cache/ \
  --global_file_segment_size=85899345920