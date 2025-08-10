#!/bin/bash

# PYTHONPATH를 설정하고 Python 스크립트를 실행하는 헬퍼 스크립트
export PYTHONPATH="/app:$PYTHONPATH"
python "$@"