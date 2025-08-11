# Conda Environment Setup Issue Resolution

## Problem Overview

Docker 컨테이너 환경에서 conda environment가 활성화되었다고 표시되지만, 실제로 `which python`과 `which pip`는 base 환경의 경로를 가리키는 문제가 발생했습니다. 이는 Claude Code와 같은 비대화형 세션과 사용자 터미널의 대화형 세션 모두에서 pytorch-legacy 환경을 자동으로 활성화하려고 할 때 나타났습니다.

## Symptoms

### 문제 증상들
1. **환경 표시 불일치**: `$CONDA_DEFAULT_ENV`는 `pytorch-legacy`로 표시되지만, `which python`은 `/opt/conda/bin/python` (base 환경) 가리킴
2. **PATH 순서 문제**: conda activate 실행 후에도 base 환경 경로가 가상환경 경로보다 우선순위가 높음
3. **세션별 다른 동작**: 대화형 세션과 비대화형 세션에서 conda 초기화가 다르게 작동
4. **PATH 중복**: `/opt/conda/bin`이 PATH에 중복으로 나타남

### 예상 vs 실제 결과
```bash
# 예상 결과
$ conda activate pytorch-legacy
$ which python
/opt/conda/envs/pytorch-legacy/bin/python

# 실제 결과  
$ conda activate pytorch-legacy
$ which python
/opt/conda/bin/python  # base 환경
```

## Root Cause Analysis

### 1. PATH 중복 문제
conda 초기화 과정에서 `/opt/conda/bin`이 여러 번 PATH에 추가되어 중복 발생:
```bash
/opt/conda/bin:/opt/conda/envs/pytorch-legacy/bin:/opt/conda/bin:/opt/conda/condabin
```

### 2. conda activate PATH 조작 방식
conda activate는 현재 PATH를 기반으로 새 환경 경로를 추가하는데, 이미 base 환경 경로가 앞에 있으면 그 순서를 유지함:
- conda activate 전: `/opt/conda/bin:...`
- conda activate 후: `/opt/conda/bin:.../opt/conda/envs/pytorch-legacy/bin:...`

### 3. zshenv와 zshrc 간 변수명 불일치
- zshenv: `_SKIP_RC_CONDA_INIT=1` 설정
- zshrc: `_CONDARC_SKIP` 확인
- 변수명이 달라서 중복 초기화 발생

### 4. 세션 타입별 처리 로직 부재
대화형/비대화형 세션에서 conda 초기화가 독립적으로 실행되어 설정 충돌 발생

## Solution Implementation

### 1. PATH 정리 함수 구현
conda 관련 경로만 선택적으로 제거하는 함수 작성:

```bash
# PATH에서 conda 관련 경로 정리 (중복 제거)
__clean_conda_path() {
  # conda 관련 경로들을 제거하고 다른 경로들만 유지 (호환성 개선)
  local clean_path=""
  local old_ifs="$IFS"
  IFS=':'
  set -- $PATH
  IFS="$old_ifs"
  
  for path_entry; do
    case "$path_entry" in
      /opt/conda|/opt/conda/*) ;;  # skip only /opt/conda paths specifically
      *) 
        if [ -z "$clean_path" ]; then
          clean_path="$path_entry"
        else
          clean_path="$clean_path:$path_entry"
        fi
        ;;
    esac
  done
  export PATH="$clean_path"
}
```

### 2. PATH 순서 보정 로직
conda activate 후 환경별 경로를 맨 앞으로 이동:

```bash
# PATH 순서 보정 (환경이 base보다 우선되도록) - sed 없이 구현
if [ -d "/opt/conda/envs/$TARGET_ENV/bin" ]; then
  # 현재 환경 경로를 제거하고 맨 앞에 추가
  local env_bin="/opt/conda/envs/$TARGET_ENV/bin"
  local temp_path=""
  local old_ifs="$IFS"
  IFS=':'
  set -- $PATH
  IFS="$old_ifs"
  
  for path_part; do
    if [ "$path_part" != "$env_bin" ]; then
      if [ -z "$temp_path" ]; then
        temp_path="$path_part"
      else
        temp_path="$temp_path:$path_part"
      fi
    fi
  done
  export PATH="$env_bin:$temp_path"
fi
```

### 3. zshenv 최적화 (비대화형 세션용)
```bash
# ~/.zshenv
# 비대화형에서만 자동 활성화 (클로드 코드 같은 케이스)
if [[ $- != *i* ]] && [ -z "$NO_AUTO_CONDA" ]; then
  # PATH 정리
  __clean_conda_path
  
  # conda 초기화
  if [ -r "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    . "$CONDA_BASE/etc/profile.d/conda.sh"
  else
    eval "$("$CONDA_BASE/bin/conda" shell.zsh hook 2>/dev/null)" || true
  fi

  # 환경 활성화
  TARGET_ENV="pytorch-legacy"
  conda activate "$TARGET_ENV" >/dev/null 2>&1 || true
  
  # PATH 순서 보정
  [PATH 순서 보정 로직]

  # 비대화형에서 처리했음을 표시
  export _NONINTERACTIVE_CONDA_DONE=1
fi
```

### 4. zshrc 최적화 (대화형 세션용)
```bash
# ~/.zshrc
# PATH 정리 및 환경 설정 (대화형 세션용)
if [[ $- == *i* ]] && [ -z "$NO_AUTO_CONDA" ]; then
  # PATH 정리
  [PATH 정리 로직]
  
  # 비대화형에서 conda가 초기화되지 않은 경우에만 초기화
  if [ -z "$_NONINTERACTIVE_CONDA_DONE" ]; then
    [conda 초기화 로직]
  fi

  # 환경 활성화 및 PATH 순서 보정
  [환경 활성화 및 PATH 보정 로직]
fi
```

## Key Technical Improvements

### 1. 시스템 명령어 의존성 제거
- **Before**: `sed` 명령어 사용으로 시스템 의존성 존재
- **After**: 순수 shell script로 구현하여 호환성 향상

### 2. 정확한 PATH 매칭
- **Before**: `*/opt/conda*` 패턴으로 과도하게 넓은 매칭
- **After**: `/opt/conda|/opt/conda/*` 정확한 패턴으로 conda 경로만 선택적 제거

### 3. 세션 타입별 최적화
- **Before**: 모든 세션에서 동일한 초기화 로직
- **After**: 대화형/비대화형 세션에 맞는 최적화된 처리

### 4. 환경 변수 통일
- **Before**: 서로 다른 변수명으로 상태 관리
- **After**: 일관된 변수명과 명확한 상태 관리

## Follow-up Issues & Resolution

### 추가로 발견된 문제들

#### Issue: System Command "Not Found" Errors
설정 적용 후 다음과 같은 오류들이 발생:
```bash
/tmp/root-code-zsh/.zshrc:119: command not found: sed
prompt_status:9: command not found: wc
```

#### Root Cause Analysis
최초 PATH 정리 로직이 너무 공격적이어서 conda 관련 경로뿐만 아니라 필수 시스템 경로까지 제거:
- **Before**: `/opt/conda*` 패턴으로 과도하게 넓은 매칭
- **Result**: `/usr/bin`, `/bin` 등 시스템 경로가 일부 상황에서 누락

#### Final Solution: Conservative PATH Management
```bash
# 개선된 PATH 정리 로직 - 보수적 접근
__clean_conda_path() {
  # conda 중복 경로만 제거하고, 필수 시스템 경로는 보존
  local clean_path=""
  local seen_paths=""
  local old_ifs="$IFS"
  IFS=':'
  set -- $PATH
  IFS="$old_ifs"
  
  for path_entry; do
    # 빈 경로는 건너뛰기
    [ -z "$path_entry" ] && continue
    
    # 이미 본 경로는 건너뛰기 (중복 제거)
    case ":$seen_paths:" in
      *":$path_entry:"*) continue ;;
    esac
    
    # 경로 추가 (모든 유효 경로 보존)
    if [ -z "$clean_path" ]; then
      clean_path="$path_entry"
    else
      clean_path="$clean_path:$path_entry"
    fi
    seen_paths="$seen_paths:$path_entry"
  done
  export PATH="$clean_path"
}
```

## Verification Results

### 최종 테스트 결과
```bash
=== 🎯 최종 완성 테스트 ===

1. 비대화형 세션 (Claude Code 환경):
Conda: pytorch-legacy
/opt/conda/envs/pytorch-legacy/bin/python
/opt/conda/envs/pytorch-legacy/bin/pip

2. 대화형 세션 (사용자 터미널 환경):
테스트 중...
/usr/bin/wc    ✅ 정상 작동
/usr/bin/sed   ✅ 정상 작동
/opt/conda/envs/pytorch-legacy/bin/python
/opt/conda/envs/pytorch-legacy/bin/pip

✅ 결과 분석:
- 두 세션 모두 pytorch-legacy 환경에서 정상 작동
- 시스템 명령어(wc, sed) 정상 작동
- python과 pip가 올바른 가상환경 경로 가리킴
```

### 성능 개선
- **대화형/비대화형 세션 모두**: ✅ pytorch-legacy 환경 자동 활성화
- **PATH 순서**: ✅ 가상환경이 base 환경보다 우선순위 높음  
- **중복 제거**: ✅ conda 경로 중복 해결
- **시스템 호환성**: ✅ 외부 명령어 의존성 제거
- **시스템 명령어**: ✅ wc, sed 등 필수 명령어 정상 작동
- **Oh My Zsh 테마**: ✅ agnoster 테마 오류 해결

## Best Practices

### 1. Conda Environment 자동화 설정 시 고려사항
- PATH 중복 방지를 위한 사전 정리 필수
- 세션 타입별 다른 초기화 로직 적용
- 환경별 경로 우선순위 명시적 관리

### 2. Docker Container에서 Conda 사용 시
- 컨테이너 빌드 시점과 런타임 환경 설정 분리
- 다양한 세션 타입에서 일관된 환경 제공
- 시스템 명령어 의존성 최소화

### 3. Shell 설정 파일 관리
- `.zshenv`: 비대화형 세션 처리
- `.zshrc`: 대화형 세션 처리  
- 명확한 책임 분리와 상태 변수 관리

## References

- [Conda Documentation - Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Zsh Documentation - Startup/Shutdown Files](http://zsh.sourceforge.net/Intro/intro_3.html)
- [PATH Environment Variable Best Practices](https://en.wikipedia.org/wiki/PATH_(variable))

---

**Created**: 2025-01-11  
**Resolution Status**: ✅ Resolved  
**Verified On**: Docker container with zsh, conda, pytorch-legacy environment