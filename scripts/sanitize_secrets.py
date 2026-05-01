#!/usr/bin/env python3
"""Sanitize hardcoded API keys in shell/python files.

Replaces:
  export OPENAI_API_KEY="sk-proj-..."  →  export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY in your env or .env}"
  (same for HF_TOKEN, GEMINI_API_KEY, TOGETHER_API_KEY, MEM0_API_KEY, ZEP_API_KEY)

Targets a list of files. Idempotent — already-sanitized lines are left alone.
"""
import re
import sys

# Each tuple: (env_var_name, regex_to_match_value)
SECRETS = [
    ("OPENAI_API_KEY",   r'sk-proj-[a-zA-Z0-9_\-]{40,}'),
    ("HF_TOKEN",         r'hf_[a-zA-Z0-9]{20,}'),
    ("GEMINI_API_KEY",   r'AIzaSy[a-zA-Z0-9_\-]{20,}'),
    ("TOGETHER_API_KEY", r'[a-f0-9]{64}(,[a-f0-9]{64})*'),
    ("MEM0_API_KEY",     r'm0-[a-zA-Z0-9_\-]{20,}'),
    ("ZEP_API_KEY",      r'z_[a-zA-Z0-9_\-\.]{40,}'),
]

def sanitize_file(path):
    try:
        with open(path) as f:
            text = f.read()
    except (FileNotFoundError, UnicodeDecodeError):
        return False, 0

    original = text
    n_subs = 0
    for env_name, pat in SECRETS:
        # Match: export VAR="<secret>" or export VAR='<secret>' or VAR="<secret>"
        regex = re.compile(
            rf'(?:export\s+)?{re.escape(env_name)}\s*=\s*["\']?({pat})["\']?'
        )
        def repl(m):
            nonlocal n_subs
            n_subs += 1
            return f'export {env_name}="${{{env_name}:?Set {env_name} via env or sourced .env file}}"'
        text = regex.sub(repl, text)

        # Also handle inline mentions (in comments)
        # `# another : <secret> - <secret>` → comment with [REDACTED]
        comment_pat = re.compile(rf'#[^\n]*({pat})[^\n]*')
        def comment_repl(m):
            nonlocal n_subs
            n_subs += 1
            return re.sub(pat, '[REDACTED]', m.group(0))
        text = comment_pat.sub(comment_repl, text)

    if text != original:
        with open(path, 'w') as f:
            f.write(text)
        return True, n_subs
    return False, 0


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print("Usage: sanitize_secrets.py <file1> <file2> ...", file=sys.stderr)
        sys.exit(1)
    total_changed = 0
    total_subs = 0
    for f in files:
        changed, n = sanitize_file(f)
        if changed:
            print(f"  ✓ {f} ({n} substitutions)")
            total_changed += 1
            total_subs += n
        else:
            print(f"    {f} (no change)")
    print(f"\nDone: {total_changed} files modified, {total_subs} total substitutions.")
