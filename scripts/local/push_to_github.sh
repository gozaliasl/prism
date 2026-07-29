#!/usr/bin/env bash

# Safe shell settings
set -euo pipefail

# Ensure git is available
if ! command -v git >/dev/null 2>&1; then
  echo "git is required but not found in PATH."
  exit 1
fi

# Verify we are inside a Git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  cat <<'EOF'
Repository not initialized.
Run `git init`, add a remote (e.g. `git remote add origin <url>`),
and then re-run this script.
EOF
  exit 1
fi

# Move to repository root
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Ensure large directories stay untracked
IGNORE_FILE="$REPO_ROOT/.gitignore"
ensure_ignore_entry() {
  local entry="$1"
  if ! grep -Fxq "$entry" "$IGNORE_FILE"; then
    echo "$entry" >> "$IGNORE_FILE"
  fi
}

touch "$IGNORE_FILE"
ensure_ignore_entry "data/"
ensure_ignore_entry "outputs/"
ensure_ignore_entry "docs/jwst_slsim_paper/"

# Stage .gitignore updates if any
git add "$IGNORE_FILE"

# Determine current branch
CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || git rev-parse --short HEAD)"

# Pull latest changes when an upstream branch is configured
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git pull --rebase --autostash
else
  echo "No upstream branch configured for $CURRENT_BRANCH. Skipping pull."
fi

# Stage tracked changes, respecting ignore rules
# Explicitly exclude paper draft, outputs, and data directories
git add --all
git reset -- docs/jwst_slsim_paper/ outputs/ data/ 2>/dev/null || true

# Abort if nothing to commit
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  COMMIT_MESSAGE="${1:-"Update $(date +'%Y-%m-%d %H:%M:%S')"}"
  git commit -m "$COMMIT_MESSAGE"
fi

# Push if remote exists
if git remote get-url origin >/dev/null 2>&1; then
  git push origin "$CURRENT_BRANCH"
else
  echo "No remote named 'origin' found. Add one with:"
  echo "  git remote add origin <git@github.com:user/repo.git>"
fi

