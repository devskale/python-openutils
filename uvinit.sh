#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
set -u

MODE=""
PACKAGE_FILTER=""
SILENT=0
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$ROOT_DIR/uvinit.log"
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a SUCCESS_LIST=()
declare -a FAIL_LIST=()
declare -a SKIP_LIST=()

usage() {
  echo "Usage: $0 [-x|-u|-c|-h] [package] [-s]"
  echo "  -x            Init packages (uv sync)"
  echo "  -u            Upgrade packages (uv lock -U, uv sync)"
  echo "  -c            Remove .venv for matched packages"
  echo "  -h            Show help"
  echo "  [package]     Optional substring to filter packages by directory name"
  echo "  -s            Silent mode (no prompts, concise output)"
  echo "  --extra NAME   Pass --extra NAME to uv sync (repeatable)"
  echo ""
  echo "Packages are auto-discovered from subdirectories containing pyproject.toml."
  echo ""
  echo "Examples:"
  echo "  $0 -x                  Init all packages"
  echo "  $0 -u                  Upgrade all packages (lock + sync)"
  echo "  $0 -x credgoo          Init packages matching 'credgoo'"
  echo "  $0 -c                  Clean all venvs"
}

log_init() {
  : > "$LOG_FILE"
  echo "uvinit start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
  local branch commit date tag
  if branch=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null); then
    commit=$(git -C "$ROOT_DIR" log -1 --oneline 2>/dev/null)
    date=$(git -C "$ROOT_DIR" log -1 --format='%ci' 2>/dev/null)
    tag=$(git -C "$ROOT_DIR" describe --tags --exact-match 2>/dev/null || true)
    {
      echo "Branch : $branch"
      echo "Commit : $commit"
      echo "Date   : $date"
      [ -n "$tag" ] && echo "Tag    : $tag"
    } | tee -a "$LOG_FILE"
  fi
}

log_info() {
  if [ "$SILENT" -eq 1 ]; then
    echo "[INFO] $1" >> "$LOG_FILE"
  else
    echo "[INFO] $1" | tee -a "$LOG_FILE"
  fi
}

log_warn() {
  if [ "$SILENT" -eq 1 ]; then
    echo "[WARN] $1" >> "$LOG_FILE"
  else
    echo "[WARN] $1" | tee -a "$LOG_FILE"
  fi
}

log_error() {
  if [ "$SILENT" -eq 1 ]; then
    echo "[ERROR] $1" >> "$LOG_FILE"
  else
    echo "[ERROR] $1" | tee -a "$LOG_FILE"
  fi
}

confirm() {
  [ "$SILENT" -eq 1 ] && return 0
  printf "%s [y/N]: " "$1"
  read -r ans
  case "$ans" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

require_uv() {
  command -v uv >/dev/null 2>&1 || { log_error "uv not found. Install with: pipx install uv or brew install uv"; exit 127; }
}

discover_projects() {
  local root="$1" results=() dirs=() unique=() seen=""
  while IFS= read -r -d '' f; do
    results+=("$f")
  done < <(find "$root" -type f \( -name "pyproject.toml" -o -name "setup.py" \) \
    -not -path "*/.venv/*" -not -path "*/venv/*" -not -path "*/site-packages/*" -not -path "*/node_modules/*" -not -path "*/scaffolding/*" -print0)
  for f in "${results[@]}"; do
    dirs+=("$(dirname "$f")")
  done
  for d in "${dirs[@]}"; do
    [[ "$d" == *"/site-packages"* ]] && continue
    [[ -n "$PACKAGE_FILTER" ]] && [[ "$d" != *"$PACKAGE_FILTER"* ]] && continue
    [[ ":$seen:" == *":$d:"* ]] && continue
    unique+=("$d")
    seen="$seen:$d"
  done
  [ "${#unique[@]}" -gt 0 ] && echo "${unique[@]}"
}

detect_python_bin() {
  local dir="$1"
  for p in "$dir/.venv/Scripts/python.exe" "$dir/.venv/Scripts/python" "$dir/.venv/bin/python" "$dir/.venv/bin/python.exe"; do
    [ -x "$p" ] && echo "$p" && return
  done
}

get_installed_version() {
  local dir="$1" py pkg_name ver
  py="$(detect_python_bin "$dir")"
  [ -z "$py" ] && return
  if [ -f "$dir/pyproject.toml" ]; then
    pkg_name=$(grep -E '^\s*name\s*=' "$dir/pyproject.toml" | head -1 \
      | sed -E 's/.*name[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/' | tr -d '[:space:]')
  fi
  [ -z "$pkg_name" ] && pkg_name="$(basename "$dir")"
  ver=$("$py" -m pip show "$pkg_name" 2>/dev/null | grep -E '^Version:' | awk '{print $2}')
  if [ -n "$ver" ]; then
    echo "$pkg_name==$ver"
  else
    local src_ver
    src_ver=$(grep -E '^\s*version\s*=' "$dir/pyproject.toml" 2>/dev/null | head -1 \
      | sed -E 's/.*version[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/' | tr -d '[:space:]')
    [ -n "$src_ver" ] && echo "$pkg_name@$src_ver" || echo "$pkg_name"
  fi
}

process_dirs() {
  local discovered
  discovered="$(discover_projects "$ROOT_DIR")"
  if [ -z "$discovered" ]; then
    log_warn "No packages found under $ROOT_DIR"; exit 2
  fi

  local -a dirs
  IFS=' ' read -r -a dirs <<< "$discovered"

  for d in "${dirs[@]}"; do
    case "$MODE" in
      init)
        log_info "Init: $d"
        if (cd "$d" && uv sync $UV_EXTRAS >>"$LOG_FILE" 2>&1); then
          log_info "Synced: $d"
          SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
        else
          log_error "uv sync failed: $d"
          FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
        fi
        ;;
      upgrade)
        log_info "Upgrade: $d"
        if ! (cd "$d" && uv lock -U >>"$LOG_FILE" 2>&1); then
          log_error "uv lock -U failed: $d"
          FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
          continue
        fi
        log_info "Lock updated: $d"
        if (cd "$d" && uv sync $UV_EXTRAS >>"$LOG_FILE" 2>&1); then
          log_info "Synced: $d"
          SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
        else
          log_error "uv sync failed: $d"
          FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
        fi
        ;;
      clean)
        log_info "Clean: $d"
        if [ ! -d "$d/.venv" ]; then
          log_warn "No venv: $d"
          SKIP_COUNT=$((SKIP_COUNT+1)); SKIP_LIST+=("$d")
          continue
        fi
        if ! confirm "Remove venv in $d?"; then
          log_warn "Skipped: $d"
          SKIP_COUNT=$((SKIP_COUNT+1)); SKIP_LIST+=("$d")
          continue
        fi
        rm -rf "$d/.venv"
        log_info "Removed: $d"
        SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
        ;;
    esac
  done
}

summary() {
  echo "----------- Summary -----------" | tee -a "$LOG_FILE"
  echo "Success: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
  if [ "$SUCCESS_COUNT" -gt 0 ]; then
    for s in "${SUCCESS_LIST[@]}"; do
      echo "  OK  - $s  ($(get_installed_version "$s"))" | tee -a "$LOG_FILE"
    done
  fi
  echo "Failed: $FAIL_COUNT" | tee -a "$LOG_FILE"
  if [ "$FAIL_COUNT" -gt 0 ]; then
    for f in "${FAIL_LIST[@]}"; do echo "  ERR - $f" | tee -a "$LOG_FILE"; done
  fi
  echo "Skipped: $SKIP_COUNT" | tee -a "$LOG_FILE"
  if [ "$SKIP_COUNT" -gt 0 ]; then
    for k in "${SKIP_LIST[@]}"; do echo "  SKP - $k" | tee -a "$LOG_FILE"; done
  fi
  [ "$FAIL_COUNT" -gt 0 ] && exit 3
  exit 0
}

parse_args() {
  [ "$#" -eq 0 ] && { usage; exit 1; }
  while getopts ":xuchs" opt; do
    case "$opt" in
      x) MODE="init" ;;
      u) MODE="upgrade" ;;
      c) MODE="clean" ;;
      h) usage; exit 0 ;;
      s) SILENT=1 ;;
      \?) usage; exit 1 ;;
    esac
  done
  shift $((OPTIND -1))
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -s) SILENT=1; shift ;;
      --extra) [ "$#" -ge 2 ] && { UV_EXTRAS="$UV_EXTRAS --extra $2"; shift 2; } || shift ;;
      *) PACKAGE_FILTER="$1"; shift ;;
    esac
  done
  [ -z "$MODE" ] && MODE="init"
}

main() {
  parse_args "$@"
  log_init
  require_uv
  log_info "Mode: $MODE, Filter: '${PACKAGE_FILTER}', Silent: $SILENT"
  process_dirs
  summary
}

main "$@"
