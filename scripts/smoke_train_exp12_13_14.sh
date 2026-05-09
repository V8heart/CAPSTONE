#!/usr/bin/env bash
# 호환용 래퍼: exp11까지 포함한 스크립트로 위임합니다.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/smoke_train_exp11_12_13_14.sh" "$@"
