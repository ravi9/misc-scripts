---
name: update-openvino-version
description: Bump the pinned OpenVINO toolkit version across the OpenVINO backend's CI workflows, Docker image, and docs, in a fresh branch off dev_backend_openvino, and verify the result builds. Use when the user wants to update/upgrade the OpenVINO version used by llama.cpp.
---

# Update the pinned OpenVINO version

Every consumer (CI, Docker, the docs' install scripts) downloads a specific OpenVINO toolkit archive from `storage.openvinotoolkit.org`, pinned by two variables that always travel together:

- `OPENVINO_VERSION_MAJOR` - the release directory on the CDN, e.g. `2026.3.1`. Also the install directory and release-artifact name. **Not "the major version"** - it's whatever directory the CDN uses, full three-part number for patch releases.
- `OPENVINO_VERSION_FULL` - the full build string, e.g. `2026.3.1.22476.56d9685302d`. Also the cache key.

URL shape both feed:
```
https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION_MAJOR}/<os>/openvino_toolkit_<flavor>_${OPENVINO_VERSION_FULL}_<arch>.<ext>
```

Job: branch off `dev_backend_openvino` -> resolve the pair -> verify archives exist -> replace the pair everywhere -> check knock-on effects -> build to confirm -> re-verify. No source-code changes here - a bump needing C++ changes is a separate commit.

Read `AGENTS.md` first if not already in context. Never commit, push, or open a PR without explicit per-action approval.

## Step 0 - Branch

The user typically invokes this by pasting the target archive URL. Read `OPENVINO_VERSION_MAJOR`/`FULL` off it (Step 1) before naming the branch.

```sh
git clone https://github.com/ravi9/llama.cpp.git
cd llama.cpp
git checkout dev_backend_openvino
git checkout -b ov-<OPENVINO_VERSION_MAJOR>
```

Branch name is `ov-` + the literal `OPENVINO_VERSION_MAJOR` value, unshortened - `ov-2026.3.1` for major `2026.3.1`, `ov-2026.4` for major `2026.4`. **If that branch already exists** (locally or on the fork), stop and ask the user for a name instead of reusing, overwriting, or improvising one.

If already cloned, `cd` in, confirm it's based on `dev_backend_openvino`, and skip re-cloning.

## Step 1 - Resolve the version pair

Given a full URL: read `OPENVINO_VERSION_MAJOR` off the `packages/<X>/` segment and `OPENVINO_VERSION_FULL` off the filename. Confirm MAJOR against the URL rather than assuming two-part form - it's often three parts.

Given only a release number (e.g. "2026.4"), find the build string:
```sh
curl -s https://storage.openvinotoolkit.org/filetree.json \
  | grep -o 'openvino_toolkit_ubuntu24_2026\.4[^"]*\.tgz' | sort -u
```
Grep for a narrow prefix (the index is ~5MB). Skip `.dev<date>` names (nightlies) unless asked for one; pick the release build (numeric build id + commit hash).

A pasted URL may use `ubuntu22` even though the repo pins `ubuntu24` - take the version from it, but keep the repo's flavor (see Gotchas).

## Step 2 - Verify archives exist (before editing anything)

Check both flavors the repo actually consumes, using the repo's pinned flavor:
```sh
V_MAJOR=2026.4
V_FULL=2026.4.0.12345.abcdef01234
for u in linux/openvino_toolkit_ubuntu24_${V_FULL}_x86_64.tgz \
         windows/openvino_toolkit_windows_${V_FULL}_x86_64.zip; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -L -r 0-0 \
    "https://storage.openvinotoolkit.org/repositories/openvino/packages/${V_MAJOR}/$u")
  echo "$code  $u"
done
```
`200`/`206` = present (use the range-request form; `curl -I` can be proxy-masked). If either is missing, stop and report it - don't bump the rest.

## Step 3 - Find every pin

Grep, don't rely on a fixed file list - it drifts:
```sh
grep -rn "OPENVINO_VERSION_MAJOR\|OPENVINO_VERSION_FULL\|<old version>\|<old build hash>" \
  --exclude-dir=.git .
```
Known locations, several with **multiple occurrences per file** (one per CI job) - replace all:
- `.devops/openvino.Dockerfile` - `ARG` defaults
- `.github/workflows/build-openvino.yml`, `build-cache.yml`, `build-self-hosted.yml`, `release.yml` - per-job `env` (Linux and Windows)
- `docs/backend/OPENVINO.md` - Linux/Windows install scripts plus prose

Each workflow `env` block carries a `# Sync versions in ...` comment naming the canonical file set - cross-check it against the grep, but trust the grep if the two disagree.

## Step 4 - Replace

Order: `OPENVINO_VERSION_FULL` (unique string) -> `OPENVINO_VERSION_MAJOR` (anchor to the variable name so unrelated numbers aren't touched) -> prose mentions in docs.

Leave alone: `docs.openvino.ai/2026/...` links (doc channel, bump only on 2026->2027), `NPU_DRIVER_VERSION`/`NPU_DRIVER_FULL` (independent driver version), the `ubuntu24` flavor, and the `e.g.` version examples in `.github/actions/{linux,windows}-setup-openvino/action.yml` (those actions take the version as an input - the examples are comments, not pins, and refreshing them just adds diff noise).

## Step 5 - Check knock-on effects of a MAJOR shape change

If MAJOR's shape changed (e.g. `2026.3` -> `2026.3.1`), confirm and tell the user:
- Install dir `/opt/intel/openvino_${MAJOR}` (Linux) / `C:\Intel\openvino_%MAJOR%` (Windows) - `setupvars` goes through the `openvino` symlink/junction, stays valid.
- Release artifact names in `release.yml` (`llama-<tag>-bin-{ubuntu,win}-openvino-<MAJOR>-x64.*`) - downstream consumers see this.
- Cache keys are on `FULL`, so they self-invalidate - no manual purge.

Also: `ggml/src/ggml-openvino/CMakeLists.txt` has no version floor, usually nothing to change; skim release notes for API removals affecting `ggml-openvino` and flag (don't fix) any needed code change.

## Step 6 - Build

Cheapest way to catch a MAJOR/FULL mismatch or missed pin before calling it done.

1. Get the toolkit in place, then source `setupvars`. If `/opt/intel/openvino_${MAJOR}` already exists, check it is the target build (`cat /opt/intel/openvino_${MAJOR}/runtime/version.txt` prints e.g. `2026.3.1-22476-56d9685302d`) and just `source /opt/intel/openvino/setupvars.sh` - no need to re-run the installer. Otherwise run the docs' own install script (now edited) to unpack it.
2. Configure and build:
   ```sh
   cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
   cmake --build build/ReleaseOV --parallel
   ```
3. Clean build = pass. Check the configure log names the new version - `Found OpenVINO: /opt/intel/openvino_<MAJOR>/runtime/cmake (found version "<MAJOR>")`. On failure, determine version-pin mistake vs. genuine API break before touching anything else.

No native toolchain? Fall back to Docker (also exercises the real download path):
```sh
docker build -f .devops/openvino.Dockerfile --target full -t llama-openvino-test .
```
Report which path you used and whether it passed. If you skip building, say so and state what's unverified.

## Step 7 - Re-verify

1. Re-grep the old version/hash - remaining hits should only be intentional (historical notes, unrelated vendored files).
2. Confirm the new pair is consistent everywhere - no new FULL paired with an old MAJOR.
3. Hand-reconstruct one URL from the edited files and resolve it (Step 2's command) - catches mismatches grep can't.

## Step 8 - Hand off

Show `git diff` and stop. Committing and pushing are the user's, per `AGENTS.md`.

Read the diff for files you did not touch before handing it over. A bump touches only the pin files, so anything else is someone else's in-flight edit - name it explicitly instead of reverting it, because `git add -u` will sweep it into their commit.

## Gotchas

- **`ubuntu24` is deliberate** - matches the Docker base image and CI runners. A pasted link may say `ubuntu22`; keep the repo's flavor regardless, and say so. Change it only alongside the Dockerfile base image and `runs-on` values.
- **Windows and Linux share one `FULL`** - bump both jobs together or their CI diverges.
- **MAJOR ≠ major version** - it's the CDN's literal directory name.
- **Verify before editing** - a missing archive found after a full sweep means reverting everything.
- **Branch name uses the full MAJOR value**, not a shortened one - don't truncate it when the branch name happens to look shorter.
