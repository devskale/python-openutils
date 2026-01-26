#!/usr/bin/env bash
set -u

MODE=""
PACKAGE_FILTER=""
SILENT=0
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$ROOT_DIR/uvinit.log"
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a SUCCESS_LIST
declare -a FAIL_LIST
declare -a SKIP_LIST
ALL_DIRS=("packages/ofs" "packages/pdf2md.skale" "packages/robotni/robotni_arq" "packages/credgoo" "packages/uniinfer" "packages/agentos" "packages/md2blank" "packages/md2pdfs")
DISCOVER=0

usage() {
  echo "Usage: $0 [-x|-v|-p|-u|-c|-h] [package] [-s|-i|-D]"
  echo "  -x            Initialize projects with uv (discover, venv, install, activate)"
  echo "  -v            Create venvs with uv for matched projects"
  echo "  -p            Install matched projects into their venvs via 'uv pip install .'"
  echo "  -u            Upgrade installs via 'uv pip install -U .' and -U on requirements"
  echo "  -c            Remove venvs for matched projects"
  echo "  -h            Show help"
  echo "  [package]     Optional substring to filter projects by directory name"
  echo "  -s            Silent mode (no prompts, concise output)"
  echo "  -i            Interactive mode (confirm per-project)"
  echo "  -D            Use discovery mode (ignore predefined package list)"
  echo ""
  echo "Examples:"
  echo "  $0 -x                  Initialize all discovered projects with uv"
  echo "  $0 -v md2              Create venvs only for projects matching 'md2'"
  echo "  $0 -p uniinfer -s      Install uniinfer silently into its venv"
  echo "  $0 -u                  Upgrade all projects using uv pip"
  echo "  $0 -x credgoo          Initialize predefined package 'credgoo'"
  echo "  $0 -x -D               Initialize all auto-discovered projects"
}

log_init() {
  : > "$LOG_FILE"
  echo "uvinit start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
}

log_info() {
  local msg="$1"
  if [ "$SILENT" -eq 1 ]; then
    echo "[INFO] $msg" >> "$LOG_FILE"
  else
    echo "[INFO] $msg" | tee -a "$LOG_FILE"
  fi
}

log_warn() {
  local msg="$1"
  if [ "$SILENT" -eq 1 ]; then
    echo "[WARN] $msg" >> "$LOG_FILE"
  else
    echo "[WARN] $msg" | tee -a "$LOG_FILE"
  fi
}

log_error() {
  local msg="$1"
  if [ "$SILENT" -eq 1 ]; then
    echo "[ERROR] $msg" >> "$LOG_FILE"
  else
    echo "[ERROR] $msg" | tee -a "$LOG_FILE"
  fi
}

confirm() {
  local prompt="$1"
  if [ "$SILENT" -eq 1 ]; then
    return 0
  fi
  printf "%s [y/N]: " "$prompt"
  read -r ans
  case "$ans" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    log_error "uv not found. Install with: pipx install uv or brew install uv"
    exit 127
  fi
}

detect_python_bin() {
  local dir="$1"
  if [ -x "$dir/.venv/Scripts/python.exe" ]; then
    echo "$dir/.venv/Scripts/python.exe"
  elif [ -x "$dir/.venv/Scripts/python" ]; then
    echo "$dir/.venv/Scripts/python"
  elif [ -x "$dir/.venv/bin/python" ]; then
    echo "$dir/.venv/bin/python"
  elif [ -x "$dir/.venv/bin/python.exe" ]; then
    echo "$dir/.venv/bin/python.exe"
  else
    echo ""
  fi
}

detect_pip_bin() {
  local dir="$1"
  if [ -x "$dir/.venv/Scripts/pip.exe" ]; then
    echo "$dir/.venv/Scripts/pip.exe"
  elif [ -x "$dir/.venv/Scripts/pip" ]; then
    echo "$dir/.venv/Scripts/pip"
  elif [ -x "$dir/.venv/bin/pip" ]; then
    echo "$dir/.venv/bin/pip"
  elif [ -x "$dir/.venv/bin/pip.exe" ]; then
    echo "$dir/.venv/bin/pip.exe"
  else
    echo ""
  fi
}

create_activation_script() {
  local dir="$1"
  local script="$dir/activate_uv.sh"
  cat > "$script" <<'EOF'
#!/usr/bin/env bash
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$DIR/.venv/bin/activate" ]; then
  . "$DIR/.venv/bin/activate"
elif [ -f "$DIR/.venv/Scripts/activate" ]; then
  . "$DIR/.venv/Scripts/activate"
else
  echo "Activation script not found in $DIR/.venv"
  exit 1
fi
EOF
  chmod +x "$script"
}

discover_projects() {
  local root="$(pwd)"
  local results=()
  while IFS= read -r -d '' f; do
    results+=("$f")
  done < <(find "$root" -type f \( -name "pyproject.toml" -o -name "setup.py" \) \
    -not -path "*/.venv/*" -not -path "*/venv/*" -not -path "*/site-packages/*" -not -path "*/node_modules/*" -print0)
  local dirs=()
  for f in "${results[@]}"; do
    local d
    d="$(dirname "$f")"
    dirs+=("$d")
  done
  local unique=()
  local seen=""
  for d in "${dirs[@]}"; do
    if [[ "$d" == *"/site-packages"* ]]; then
      continue
    fi
    if [[ -n "$PACKAGE_FILTER" ]] && [[ "$d" != *"$PACKAGE_FILTER"* ]]; then
      continue
    fi
    if [[ ":$seen:" != *":$d:"* ]]; then
      unique+=("$d")
      seen="$seen:$d"
    fi
  done
  if [ "${#unique[@]}" -gt 0 ]; then
    echo "${unique[@]}"
  else
    echo ""
  fi
}

resolve_dirs() {
  local dirs=()
  if [ "$DISCOVER" -eq 1 ]; then
    IFS=' ' read -r -a dirs <<<"$(discover_projects)"
  else
    if [ -n "$PACKAGE_FILTER" ]; then
      local matched=()
      for dir in "${ALL_DIRS[@]}"; do
        if [[ "$dir" == *"$PACKAGE_FILTER"* ]]; then
          matched=("$dir")
          break
        fi
      done
      if [ "${#matched[@]}" -eq 0 ]; then
        matched=("packages/$PACKAGE_FILTER")
      fi
      dirs=("${matched[@]}")
    else
      dirs=("${ALL_DIRS[@]}")
    fi
  fi
  if [ "${#dirs[@]}" -gt 0 ]; then
    echo "${dirs[@]}"
  else
    echo ""
  fi
}

create_venv() {
  local dir="$1"
  if [ -d "$dir/.venv" ]; then
    log_info "venv exists: $dir"
    return 0
  fi
  if ! confirm "Create uv venv in $dir?"; then
    log_warn "Skipped venv creation: $dir"
    SKIP_COUNT=$((SKIP_COUNT+1)); SKIP_LIST+=("$dir")
    return 0
  fi
  if uv venv "$dir/.venv" >>"$LOG_FILE" 2>&1; then
    log_info "Created venv: $dir/.venv"
    return 0
  else
    log_error "Failed to create venv: $dir"
    return 1
  fi
}

install_project() {
  local dir="$1"
  local upgrade_flag="$2"
  local py
  py="$(detect_python_bin "$dir")"
  if [ -z "$py" ]; then
    log_error "Python not found in venv for $dir"
    return 1
  fi
  {
    local abs_dir
    if [[ "$dir" != /* ]]; then
      abs_dir="$ROOT_DIR/$dir"
    else
      abs_dir="$dir"
    fi
    if [[ "${py:0:1}" != "/" ]]; then
      if [[ "${py:0:2}" == "./" ]]; then
        py="$abs_dir/${py#./}"
      else
        py="$ROOT_DIR/$py"
      fi
    fi
  }
  log_info "Interpreter: $py"
  if ! uv pip install --python "$py" --upgrade pip >>"$LOG_FILE" 2>&1; then
    log_warn "pip upgrade failed in $dir"
  fi
  if [ -f "$dir/requirements.txt" ]; then
    if [ "$upgrade_flag" = "1" ]; then
      if ! (cd "$dir" && uv pip install --python "$py" -U -r requirements.txt >>"$LOG_FILE" 2>&1); then
        log_error "Requirements upgrade failed: $dir"
        return 1
      fi
    else
      if ! (cd "$dir" && uv pip install --python "$py" -r requirements.txt >>"$LOG_FILE" 2>&1); then
        log_error "Requirements install failed: $dir"
        return 1
      fi
    fi
  fi
  if [ -f "$dir/setup.py" ] || [ -f "$dir/pyproject.toml" ]; then
    if [ "$upgrade_flag" = "1" ]; then
      if ! (cd "$dir" && uv pip install --python "$py" -U . >>"$LOG_FILE" 2>&1); then
        log_error "Project upgrade install failed: $dir"
        return 1
      fi
    else
      if ! (cd "$dir" && uv pip install --python "$py" . >>"$LOG_FILE" 2>&1); then
        log_error "Project install failed: $dir"
        return 1
      fi
    fi
  else
    log_warn "No installable project file in $dir"
  fi
  return 0
}

clean_venv() {
  local dir="$1"
  if [ -d "$dir/.venv" ]; then
    if ! confirm "Remove venv in $dir?"; then
      log_warn "Skipped removal: $dir"
      SKIP_COUNT=$((SKIP_COUNT+1)); SKIP_LIST+=("$dir")
      return 0
    fi
    rm -rf "$dir/.venv"
    log_info "Removed venv: $dir"
  else
    log_warn "No venv found: $dir"
  fi
}

process_init() {
  local dirs=("$@")
  for d in "${dirs[@]}"; do
    log_info "Initializing: $d"
    if create_venv "$d"; then
      if install_project "$d" 0; then
        create_activation_script "$d"
        log_info "Initialized successfully: $d"
        SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
      else
        log_error "Initialization failed during install: $d"
        FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
      fi
    else
      log_error "Initialization failed during venv creation: $d"
      FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
    fi
  done
}

process_mode() {
  local dirs
  IFS=' ' read -r -a dirs <<<"$(resolve_dirs)"
  if [ "${#dirs[@]}" -eq 0 ]; then
    log_warn "No matching projects found"
    exit 2
  fi
  case "$MODE" in
    init)
      process_init "${dirs[@]}"
      ;;
    venv)
      for d in "${dirs[@]}"; do
        log_info "Venv: $d"
        if create_venv "$d"; then
          SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
        else
          FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
        fi
      done
      ;;
    install)
      for d in "${dirs[@]}"; do
        log_info "Install: $d"
        if [ ! -d "$d/.venv" ]; then
          log_error "Missing venv, run -v first: $d"
          FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
          continue
        fi
        if install_project "$d" 0; then
          create_activation_script "$d"
          SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
        else
          FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
        fi
      done
      ;;
    upgrade)
      for d in "${dirs[@]}"; do
        log_info "Upgrade: $d"
        if [ ! -d "$d/.venv" ]; then
          log_error "Missing venv, run -v first: $d"
          FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
          continue
        fi
        if install_project "$d" 1; then
          SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
        else
          FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
        fi
      done
      ;;
    clean)
      for d in "${dirs[@]}"; do
        log_info "Clean: $d"
        clean_venv "$d"
        SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
      done
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

parse_args() {
  if [ "$#" -eq 0 ]; then
    usage
    exit 1
  fi
  while getopts ":xvpuchsD" opt; do
    case "$opt" in
      x) MODE="init" ;;
      v) MODE="venv" ;;
      p) MODE="install" ;;
      u) MODE="upgrade" ;;
      c) MODE="clean" ;;
      h) usage; exit 0 ;;
      i) SILENT=0 ;;
      s) SILENT=1 ;;
      D) DISCOVER=1 ;;
      \?) usage; exit 1 ;;
    esac
  done
  shift $((OPTIND -1))
  if [ "$#" -gt 0 ]; then
    case "$1" in
      -s) SILENT=1; shift ;;
      *) PACKAGE_FILTER="$1"; shift ;;
    esac
  fi
  if [ "$#" -gt 0 ]; then
    if [ "$1" = "-s" ]; then
      SILENT=1
      shift
    fi
  fi
  if [ -z "$MODE" ]; then
    MODE="init"
  fi
}

summary() {
  echo "----------- Summary -----------" | tee -a "$LOG_FILE"
  echo "Success: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
  if [ "$SUCCESS_COUNT" -gt 0 ]; then
    for s in "${SUCCESS_LIST[@]}"; do echo "  OK  - $s" | tee -a "$LOG_FILE"; done
  fi
  echo "Failed: $FAIL_COUNT" | tee -a "$LOG_FILE"
  if [ "$FAIL_COUNT" -gt 0 ]; then
    for f in "${FAIL_LIST[@]}"; do echo "  ERR - $f" | tee -a "$LOG_FILE"; done
  fi
  echo "Skipped: $SKIP_COUNT" | tee -a "$LOG_FILE"
  if [ "$SKIP_COUNT" -gt 0 ]; then
    for k in "${SKIP_LIST[@]}"; do echo "  SKP - $k" | tee -a "$LOG_FILE"; done
  fi
  if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 3
  else
    exit 0
  fi
}

main() {
  parse_args "$@"
  log_init
  require_uv
  log_info "Mode: $MODE, Filter: '${PACKAGE_FILTER}', Silent: $SILENT"
  process_mode
  summary
}

main "$@"
