#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
set -u

MODE=""
PACKAGE_FILTER=""
SILENT=0
CLI_EXTRAS=""
NO_SERVICES=0
SERVICES_ONLY=0
# UVINIT_PROD env (e.g. set per prod server) defaults prod mode so ANY uvinit
# run there builds wheels — protects against a manual editable sync on prod.
# The --prod flag overrides; env is the belt-and-suspenders for deploy paths.
PROD="${UVINIT_PROD:-}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$ROOT_DIR/uvinit.log"
# This repo backs runnable services (worker) iff it ships install-services.sh.
# openutils (shared LLM infra, consumed by python-utils via git) does not — its
# one service (uniioai-proxy) is restarted by run_exec, not here. Wheel-build
# + service install are gated on SERVICES_REPO; timeout/setupoptions are always on.
# (This file is the CANONICAL uvinit — byte-identical in python-utils &
#  python-openutils via scripts/lib/uvinit-sync; edit here, re-sync.)
SERVICES_REPO=0
[ -f "$ROOT_DIR/install-services.sh" ] && SERVICES_REPO=1
# Machine-dependent setup extras (repos.yml setupoptions[machine][pkg]); best-effort,
# no-op outside the metarepo layout (no scripts/lib/repos + repos.yml found upwards).
# Machine identity is self-declared in repos.yml (per-machine) — not guessed from env.
REPOS_LIB="" REPOS_YAML_PATH="" MACHINE=""
__d="$ROOT_DIR"
while [ "$__d" != "/" ]; do
  if [ -x "$__d/scripts/lib/repos" ] && [ -f "$__d/repos.yml" ]; then
    REPOS_LIB="$__d/scripts/lib/repos"; REPOS_YAML_PATH="$__d/repos.yml"; break
  fi
  __d="$(dirname "$__d")"
done
if [ -n "$REPOS_LIB" ]; then
  MACHINE="$(REPOS_YAML="$REPOS_YAML_PATH" "$REPOS_LIB" machine 2>/dev/null)"
fi
[ -n "$MACHINE" ] || MACHINE="$(hostname -s 2>/dev/null)"   # last-resort fallback
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a SUCCESS_LIST=()
declare -a FAIL_LIST=()
declare -a SKIP_LIST=()

usage() {
  echo "Usage: $0 [-x|-u|-c|-h] [package] [-s]"
  echo "  -x            Build all packages (uv sync — install deps + build)"
  echo "  -u            Upgrade all packages (uv lock -U, uv sync)"
  echo "  -c            Remove .venv for matched packages"
  echo "  -h            Show help"
  echo "  [package]     Optional substring to filter packages by directory name"
  echo "  -s            Silent mode (no prompts, concise output)"
  echo "  --extra NAME   Pass --extra NAME to uv sync (repeatable)"
  echo "  --no-services       Skip systemd service install/restart after build"
  echo "  --services,-r       Sync + (re)install + restart services"
  echo "                      (safe one-liner after a pull — syncs venvs,"
  echo "                       smoke-verifies imports, then restarts)"
  echo "  --prod              PROD build: frozen deps (uv.lock) + built wheel"
  echo "                      (non-editable). Dev default is editable; prod must"
  echo "                      NEVER be editable (wheel shadows via PYTHONPATH gone)."
  echo ""
  echo "Packages are auto-discovered from subdirectories containing pyproject.toml."
  echo ""
  echo "Examples:"
  echo "  $0 -x                  Build all packages"
  echo "  $0 -u                  Upgrade all packages (lock + sync)"
  echo "  $0 -x credgoo          Build packages matching 'credgoo'"
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

# ── Hard failsafe: a uv call can never hang the deploy ────────────────────
# uv sync/pip/lock can stall — a slow ARM wheel build, a frozen network fetch,
# or a credential prompt under non-interactive SSH — and wedge the whole
# deploy (the SSH never returns). Redefine `uv` so EVERY call in this script
# is bounded by UV_TIMEOUT: SIGTERM at the deadline, SIGKILL 10s later
# (--kill-after). The inner `uv` is exec'd by `timeout` directly (PATH binary)
# — no function recursion. Override per run:
#   UV_TIMEOUT=600 ./uvinit.sh …  (lower for fast boxes / CI)
UV_TIMEOUT="${UV_TIMEOUT:-1800}"
_TIMEOUT_BIN=""
command -v timeout  >/dev/null 2>&1 && _TIMEOUT_BIN=timeout
command -v gtimeout >/dev/null 2>&1 && _TIMEOUT_BIN=gtimeout
uv() {
  local rc
  if [ -n "$_TIMEOUT_BIN" ]; then
    "$_TIMEOUT_BIN" --kill-after=10s "${UV_TIMEOUT}" uv "$@"
  else
    command uv "$@"          # no timeout binary (rare) — unbounded fallback
  fi
  rc=$?
  case "$rc" in
    124) log_error "uv $1 … TIMED OUT after ${UV_TIMEOUT}s (--kill-after=10s)" ;;
    137) log_error "uv $1 … SIGKILLed after timeout grace — likely OOM or a stuck build" ;;
  esac
  return "$rc"
}

# Reap child `timeout`/`uv` on exit or signal so killing uvinit (pkill uvinit,
# Ctrl-C, a deploy abort) does not orphan a uv that would otherwise be
# reparented to init and keep running. Child = the `timeout` wrapper;
# grandchild = `uv`. Grandchild first (clean SIGTERM), then the wrapper.
_reap_children() {
  local child
  for child in $(pgrep -P "$$" 2>/dev/null || true); do
    pkill -TERM -P "$child" 2>/dev/null || true
    kill  -TERM "$child"    2>/dev/null || true
  done
}
trap _reap_children EXIT INT TERM

confirm() {
  [ "$SILENT" -eq 1 ] && return 0
  printf "%s [y/N]: " "$1"
  read -r ans
  case "$ans" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

UV_MIN_VERSION="0.7.0"

require_uv() {
  command -v uv >/dev/null 2>&1 || { log_error "uv not found. Install with: pipx install uv or brew install uv"; exit 127; }
  local uv_ver
  uv_ver=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
  if ! printf '%s\n%s\n' "$UV_MIN_VERSION" "$uv_ver" | sort -V | head -1 | grep -q "$UV_MIN_VERSION"; then
    log_error "uv $uv_ver is too old (need >= $UV_MIN_VERSION). Run: uv self update"
    exit 127
  fi
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
  local dir="$1" py pkg_name ver=""
  if [ -f "$dir/pyproject.toml" ]; then
    pkg_name=$(grep -E '^\s*name\s*=' "$dir/pyproject.toml" | head -1 \
      | sed -E 's/.*name[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/' | tr -d '[:space:]')
  fi
  [ -z "$pkg_name" ] && pkg_name="$(basename "$dir")"
  py="$(detect_python_bin "$dir")"
  if [ -n "$py" ]; then
    ver=$("$py" -c "from importlib.metadata import version; print(version('$pkg_name'))" 2>/dev/null)
  fi
  if [ -n "$ver" ]; then
    echo "$pkg_name==$ver"
  else
    local src_ver
    src_ver=$(grep -E '^\s*version\s*=' "$dir/pyproject.toml" 2>/dev/null | head -1 \
      | sed -E 's/.*version[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/' | tr -d '[:space:]')
    [ -n "$src_ver" ] && echo "$pkg_name@$src_ver" || echo "$pkg_name"
  fi
}

# Extras a package actually defines — keys under [project.optional-dependencies].
defined_extras() {  # $1 = package dir → prints space-sep defined extra names
  local f="$1/pyproject.toml"
  [ -f "$f" ] || return
  awk '
    /^\[project\.optional-dependencies\]/ { inopt = 1; next }
    /^\[/ { inopt = 0 }
    inopt && /^[A-Za-z0-9_.-]+[[:space:]]*=/ { sub(/[[:space:]]*=.*/, ""); print }
  ' "$f"
}

# Per-package uv-sync extras: CLI --extra flags + machine setupoptions (repos.yml).
# FILTERED to extras the package actually defines — a blanket --extra (e.g. a
# deployto UV_EXTRAS=full meant only for pdf2md) is silently skipped for
# packages without that extra, instead of failing uv sync with
# "Extra X is not defined in the project's optional-dependencies table".
setup_extras_for() {  # $1 = package dir → prints "--extra X --extra Y …" (only defined)
  local pkgdir="$1" base pyname key opts e req defined out=""
  req="$CLI_EXTRAS"                                  # "--extra a --extra b …"
  if [ -n "$REPOS_LIB" ] && [ -n "$MACHINE" ]; then
    base="$(basename "$pkgdir")"
    pyname=$(grep -E '^[[:space:]]*name[[:space:]]*=' "$pkgdir/pyproject.toml" 2>/dev/null | head -1 \
      | sed -E 's/.*name[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/' | tr -d '[:space:]')
    for key in "$base" "$pyname" "${base%.skale}" "${pyname%-skale}"; do
      [ -z "$key" ] && continue
      opts="$(REPOS_YAML="$REPOS_YAML_PATH" "$REPOS_LIB" setupoptions "$MACHINE" "$key" 2>/dev/null)" || { opts=""; continue; }
      [ -n "$opts" ] && break
    done
    req="$req $opts"
  fi
  req=$(printf '%s' "$req" | sed -E 's/--extra[[:space:]]+//g')   # → "a b c"
  defined="$(defined_extras "$pkgdir")"
  for e in $req; do
    [ -z "$e" ] && continue
    case " $defined " in *" $e "*)
      case " $out " in *" --extra $e "*) ;; *) out="$out --extra $e" ;; esac ;;
      *) log_info "skip extra '$e' — not defined in $pkgdir" ;;
    esac
  done
  printf '%s' "$out"
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
    UV_EXTRAS="$(setup_extras_for "$d")"   # CLI extras + machine setupoptions (per package)
    case "$MODE" in
      init)
        log_info "Init: $d"
        if [ "$PROD" = 1 ]; then
          # Fail-fast lock-freshness check. A stale uv.lock (pyproject changed
          # without `uv lock`) makes `uv sync --frozen` fail with an opaque
          # error buried in uvinit.log. Check first + emit an actionable msg.
          # (`uv lock --check` is read-only — it never mutates the lock.)
          if ! (cd "$d" && uv lock --check >>"$LOG_FILE" 2>&1); then
            log_error "LOCK STALE: $d — pyproject.toml changed without 'uv lock'."
            log_error "  Fix: (cd $d && uv lock) then commit+push uv.lock, and redeploy."
            FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
            continue
          fi
          if [ "$SERVICES_REPO" = 1 ]; then
            # WORKER (backs services): frozen deps from uv.lock + built wheel
            # (non-editable). uv pip install . --no-deps replaces the editable
            # install with the wheel (removes the __editable__ .pth finder).
            # Critical: do NOT run `uv run` afterward — it re-syncs the project
            # editable and undoes this. verify_imports uses the venv python from
            # a neutral dir.
            if (cd "$d" && uv sync --frozen $UV_EXTRAS >>"$LOG_FILE" 2>&1 \
               && uv pip install . --no-deps >>"$LOG_FILE" 2>&1); then
              log_info "Synced+built (prod, non-editable): $d"
              SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
            else
              log_error "prod sync/build failed: $d"
              FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
            fi
          else
            # SHARED INFRA (consumed via git, not run as wheels): frozen only.
            if (cd "$d" && uv sync --frozen $UV_EXTRAS >>"$LOG_FILE" 2>&1); then
              log_info "Synced (prod, frozen): $d"
              SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
            else
              log_error "prod sync failed: $d"
              FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
            fi
          fi
        elif [ -n "$UV_EXTRAS" ]; then
          if (cd "$d" && uv sync $UV_EXTRAS >>"$LOG_FILE" 2>&1); then
            log_info "Synced: $d"
            SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
          elif (cd "$d" && uv sync >>"$LOG_FILE" 2>&1); then
            log_info "Synced: $d (no extras)"
            SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
          else
            log_error "uv sync failed: $d"
            FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
          fi
        else
          if (cd "$d" && uv sync >>"$LOG_FILE" 2>&1); then
            log_info "Synced: $d"
            SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
          else
            log_error "uv sync failed: $d"
            FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
          fi
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
        if [ -n "$UV_EXTRAS" ]; then
          if (cd "$d" && uv sync $UV_EXTRAS >>"$LOG_FILE" 2>&1); then
            log_info "Synced: $d"
            SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
          elif (cd "$d" && uv sync >>"$LOG_FILE" 2>&1); then
            log_info "Synced: $d (no extras)"
            SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
          else
            log_error "uv sync failed: $d"
            FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
          fi
        else
          if (cd "$d" && uv sync >>"$LOG_FILE" 2>&1); then
            log_info "Synced: $d"
            SUCCESS_COUNT=$((SUCCESS_COUNT+1)); SUCCESS_LIST+=("$d")
          else
            log_error "uv sync failed: $d"
            FAIL_COUNT=$((FAIL_COUNT+1)); FAIL_LIST+=("$d")
          fi
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
  [ "$FAIL_COUNT" -gt 0 ] && return 1
  return 0
}

verify_imports() {
  # Failsafe before disrupting running services: smoke-import each synced
  # package's top-level module. Catches stale/broken venvs that `uv sync`
  # exit-0'd but can't actually import (the tu incident: a restart-only -r
  # skipped sync, so robotni-api restarted into a venv missing llminvoke).
  # Returns 0 if all import cleanly, 1 if any fail (fail-closed: no restart).
  #
  # PROD note: verify with the VENV python from a NEUTRAL dir (cd /). `uv run`
  # would re-sync the project editable and (a) shadow the just-built wheel and
  # (b) undo the non-editable install. cd / avoids the cwd-source shadow too.
  local d mod subdir b proj_name fails=0
  log_info "Verifying imports before restart (fail-closed)…"
  for d in "${SUCCESS_LIST[@]}"; do
    # Skip non-production packages (deprecated/, scaffolding/, hello_world/) —
    # they sync but back no service, so a broken import there must not block a
    # deploy of the packages that actually run.
    case "$d" in */deprecated/*|*/scaffolding/*|*/hello_world/*) continue ;; esac
    # Workspace root (pyproject has packages = []): backs no service — its
    # children are verified individually. `common/` is a source-only shared
    # module (imported via cwd by services), NOT a built-wheel module, so
    # `import common` from a neutral dir is a false fail.
    [ "$d" = "$ROOT_DIR" ] && continue
    # Authoritative import name: the subdir matching the pyproject [project]
    # name (normalized - → _). Falls back to the first subdir with __init__.py
    # — covers pdf2md.skale → pdf2md (where name=pdf2md-skale ≠ dir ≠ module)
    # and strukt2meta (picks strukt2meta over the legacy scanflow subdir).
    proj_name=$(grep -E '^[[:space:]]*name[[:space:]]*=' "$d/pyproject.toml" 2>/dev/null | head -1 \
      | sed -E 's/.*"([^"]*)".*/\1/' | tr '-' '_')
    mod=""
    for subdir in "$d"/* "$d"/src/*; do
      [ -f "$subdir/__init__.py" ] 2>/dev/null || continue
      b="$(basename "$subdir")"
      case "$b" in tests|test|node_modules|.venv|build|dist) continue ;; esac
      [ "$b" = "$proj_name" ] && { mod="$b"; break; }   # prefer name match
      [ -z "$mod" ] && mod="$b"                            # else first candidate
    done
    [ -z "$mod" ] && mod="$(basename "$d")"
    if [ "$PROD" = 1 ] && [ "$SERVICES_REPO" = 1 ]; then
      # WORKER prod: built wheel (non-editable) — verify with the venv python
      # from a NEUTRAL dir. `uv run` would re-sync the project editable and
      # (a) shadow the just-built wheel and (b) undo the non-editable install.
      local _py; _py="$(detect_python_bin "$d")"
      if [ -n "$_py" ] && (cd / && "$_py" -c "import ${mod}" >>"$LOG_FILE" 2>&1); then
        log_info "verify OK: ${mod} (built wheel)"
      else
        log_error "verify FAILED: cannot import '${mod}' (built wheel) in ${d}"
        fails=$((fails+1))
      fi
    elif (cd "$d" && uv run python -c "import ${mod}" >>"$LOG_FILE" 2>&1); then
      log_info "verify OK: ${mod}"
    else
      log_error "verify FAILED: cannot import '${mod}' in ${d}"
      fails=$((fails+1))
    fi
  done
  [ "$fails" -eq 0 ]
}

parse_args() {
  [ "$#" -eq 0 ] && { usage; exit 1; }
  # Consume --extra flags before getopts (getopts can't handle --)
  local _args=("$@")
  local _final=()
  local _i=0
  while [ $_i -lt ${#_args[@]} ]; do
    case "${_args[$_i]}" in
      --extra)
        [ $((_i+1)) -lt ${#_args[@]} ] && { CLI_EXTRAS="$CLI_EXTRAS --extra ${_args[$((_i+1))]}"; _i=$((_i+2)); continue; } ;;
      --no-services)
        NO_SERVICES=1; _i=$((_i+1)); continue ;;
      --prod)
        PROD=1; _i=$((_i+1)); continue ;;
      --services|--restart-services|-r)
        SERVICES_ONLY=1; _i=$((_i+1)); continue ;;
    esac
    _final+=("${_args[$_i]}")
    _i=$((_i+1))
  done
  set -- "${_final[@]}"
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
      *) PACKAGE_FILTER="$1"; shift ;;
    esac
  done
  [ -z "$MODE" ] && MODE="init"
}

install_and_restart() {
  log_info "Installing & restarting systemd services..."
  bash "$ROOT_DIR/install-services.sh" install
}

main() {
  parse_args "$@"
  log_init

  # --services/-r: sync + verify + restart. Formerly a restart-only shortcut,
  # but that restarted into stale venvs after a pull (the tu incident). Now it
  # runs the full safe path. For a true restart-only (after a .service template
  # tweak with no dep change), run install-services.sh directly.
  if [ "$SERVICES_ONLY" -eq 1 ]; then
    MODE="init"
  fi

  require_uv
  log_info "Mode: $MODE, Filter: '${PACKAGE_FILTER}', Silent: $SILENT"
  process_dirs
  if ! summary; then
    exit 3
  fi
  # VERIFY imports after a successful full sync — decoupled from the restart.
  # This makes `uvinit.sh -x --no-services` a non-disruptive PREFLIGHT: it syncs
  # the new deps + smoke-imports every package, but leaves running services
  # untouched. Green preflight ⇒ the subsequent restart is safe. Fail-closed:
  # a broken venv aborts before any service is disrupted. Skipped for a single-
  # package filter (partial/dev sync).
  if [ -z "$PACKAGE_FILTER" ] && [ "$FAIL_COUNT" -eq 0 ]; then
    if ! verify_imports; then
      log_error "=========================================================="
      log_error "VERIFY FAILED — services NOT restarted."
      log_error "Running services are untouched (still on the previous code)."
      log_error "See uvinit.log for the failing import, fix it, then re-run:"
      log_error "  $0 -x"
      log_error "=========================================================="
      exit 4
    fi
    [ "$NO_SERVICES" -eq 1 ] && log_info "Preflight OK (--no-services: sync + verify done, services NOT restarted)"
  fi
  # Restart services after a successful sync + verify, unless --no-services
  # (preflight) or a single-package filter (partial/dev sync).
  if [ "$NO_SERVICES" -eq 0 ] && [ -z "$PACKAGE_FILTER" ] && [ "$FAIL_COUNT" -eq 0 ]; then
    if command -v systemctl &>/dev/null && [ -f "$ROOT_DIR/install-services.sh" ]; then
      install_and_restart
    fi
  fi
}

main "$@"
