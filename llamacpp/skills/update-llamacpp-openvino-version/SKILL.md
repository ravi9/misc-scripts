---
name: update-llamacpp-openvino-version
description: Bump the pinned OpenVINO toolkit version across the OpenVINO backend's CI workflows, Docker image, and docs. Use when the user wants to update/upgrade the OpenVINO version used by llama.cpp.
---

# Update the pinned OpenVINO version

The OpenVINO backend does not build against a system OpenVINO. Every consumer (CI, Docker, the docs' copy-paste install scripts) downloads a specific toolkit archive from `storage.openvinotoolkit.org`, pinned by two variables that always travel together:

- `OPENVINO_VERSION_MAJOR` - the release directory on the CDN, e.g. `2026.3.1`. Also names the install directory and the release artifact.
- `OPENVINO_VERSION_FULL` - the full build string, e.g. `2026.3.1.22476.56d9685302d`. Also the cache key.

The URL both feed is:

```
https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION_MAJOR}/<os>/openvino_toolkit_<flavor>_${OPENVINO_VERSION_FULL}_<arch>.<ext>
```

The whole job is: resolve the new pair, verify the archives really exist, replace the pair everywhere, then check nothing version-shaped was missed. There is no source-code change - if a bump needs C++ changes, that is a separate concern and a separate commit.

Read `AGENTS.md` before starting if it is not in context. Do not commit, push, or open a PR without explicit per-action approval from the user.

## Step 0 - Resolve the target version pair

Ask the user for the target version if they have not given it. What you need is the exact archive filename, because the build string cannot be guessed from the release number.

If the user gives a full archive URL, read both values straight off it and skip the lookup.

If the user gives only a release number (e.g. "2026.4"), find the build string. The CDN directory listing is rendered client-side and cannot be grepped, but the bucket's file index can:

```sh
curl -s https://storage.openvinotoolkit.org/filetree.json \
  | grep -o 'openvino_toolkit_ubuntu24_2026\.4[^"]*\.tgz' | sort -u
```

That index is ~5 MB, so grep for a narrow version prefix. It lists nightly builds too - names containing `.dev<date>` are nightlies, not releases. Pick the release build (a numeric build id and commit hash, e.g. `2026.4.0.12345.abcdef01234`) unless the user explicitly asked for a nightly.

Then set:
- `OPENVINO_VERSION_MAJOR` = the `packages/<X>/` directory segment
- `OPENVINO_VERSION_FULL` = the version part of the filename

**The MAJOR value is a literal path segment, not a truncated version.** For a patch release the directory is the full three-part number: `packages/2026.3.1/`, so MAJOR is `2026.3.1`, not `2026.3`. Confirm the segment against the URL rather than assuming the two-part form. Getting this wrong 404s every download.

## Step 1 - Verify the archives exist before editing anything

Check every flavor the repo consumes - Linux CI/Docker and Windows CI. Do not trust that a Windows or Linux archive exists just because the other does.

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

`200` or `206` means present. A proxy can mask the real status in `curl -I` output, so use this form (`-w "%{http_code}"` with a range request) rather than reading headers.

If any flavor is missing, stop and report which one to the user instead of bumping the rest.

## Step 2 - Find every pin

Do not work from a hardcoded file list; it drifts. Grep for the current values:

```sh
grep -rn "OPENVINO_VERSION_MAJOR\|OPENVINO_VERSION_FULL\|<old version>\|<old build hash>" \
  --exclude-dir=.git .
```

As of this writing the pins live in:

- `.devops/openvino.Dockerfile` - `ARG` defaults at the top
- `.github/workflows/build-openvino.yml` - per-job `env` (Linux and Windows jobs)
- `.github/workflows/build-cache.yml` - per-job `env`
- `.github/workflows/build-self-hosted.yml` - per-job `env`
- `.github/workflows/release.yml` - per-job `env` (Linux and Windows release jobs)
- `docs/backend/OPENVINO.md` - the Linux `bash` and Windows `cmd` install scripts, plus the prose notes that name the pinned version

Note that several workflow files pin the pair more than once, once per job. Replace all of them; a single-occurrence assumption leaves a job on the old version and the CI failure is far from the cause.

`.github/actions/linux-setup-openvino/action.yml` and `.github/actions/windows-setup-openvino/action.yml` build the URLs but take the version as an input - they hold no pin. They do carry `e.g.` example versions in their input descriptions, which are worth refreshing so they do not rot.

## Step 3 - Replace the pair

Replace `OPENVINO_VERSION_FULL` first (it is a unique string), then `OPENVINO_VERSION_MAJOR` (match it anchored to the variable name, so you do not rewrite unrelated numbers), then the prose mentions in the docs.

Leave these alone:
- `https://docs.openvino.ai/2026/...` links - a documentation channel, not a package pin. Bump only when the release moves to a new doc channel (e.g. 2026 -> 2027).
- `NPU_DRIVER_VERSION` / `NPU_DRIVER_FULL` in `.devops/openvino.Dockerfile` - the Intel NPU driver, versioned independently of OpenVINO.
- The `ubuntu24` archive flavor - see the gotcha below.

## Step 4 - Check what else the new MAJOR value touches

`OPENVINO_VERSION_MAJOR` is not only a URL segment, so changing its shape (e.g. `2026.3` -> `2026.3.1`) has visible knock-on effects. Confirm each and tell the user:

- Install directory: `/opt/intel/openvino_${OPENVINO_VERSION_MAJOR}` on Linux, `C:\Intel\openvino_%OPENVINO_VERSION_MAJOR%` on Windows. The `setupvars` path in the docs goes through the `/opt/intel/openvino` symlink (Windows: junction), so it stays valid.
- Release artifact names in `release.yml`: `llama-<tag>-bin-ubuntu-openvino-<MAJOR>-x64.tar.gz` and `llama-<tag>-bin-win-openvino-<MAJOR>-x64.zip`. Downstream consumers of these filenames will see the change.
- CI cache keys are keyed on `OPENVINO_VERSION_FULL`, so they self-invalidate. No manual cache purge needed.

Also check whether the new release needs anything beyond the pins:

- `ggml/src/ggml-openvino/CMakeLists.txt` uses a bare `find_package(OpenVINO REQUIRED ...)` with no minimum version, so no change is normally required. If the user wants a floor enforced, that is a deliberate change to discuss, not part of a routine bump.
- Skim the release notes for API removals or behavior changes that affect `ggml/src/ggml-openvino/`. If the backend needs code changes, raise it and keep it out of this commit.

## Step 5 - Verify

1. Re-grep for the old version and build hash. The only remaining hits should be intentional (e.g. historical notes, unrelated vendored files).

   ```sh
   grep -rn "<old build hash>" --exclude-dir=.git .
   ```

2. Confirm the new pair reads consistently everywhere, and that no file ended up with a new FULL next to an old MAJOR.

3. Reconstruct one URL by hand from the edited files and check it resolves, using the Step 1 command. This catches a MAJOR/FULL mismatch that grep cannot see.

4. Optional but the strongest check available locally - build the Docker image, which exercises the real download path:

   ```sh
   docker build -f .devops/openvino.Dockerfile --target full -t llama-openvino-test .
   ```

   Tell the user if you skip it, and say what remains unverified.

## Gotchas

- **Linux archive flavor is `ubuntu24`, deliberately.** The Docker base image and the CI runners are Ubuntu 24.04. OpenVINO also publishes `ubuntu22` (and other) archives, and a user pasting a download link may well paste the `ubuntu22` one. Keep the repo's `ubuntu24` pin and say so, rather than silently switching flavors. Only change it alongside the Dockerfile base image and the workflow `runs-on` values.
- **Windows and Linux share one `OPENVINO_VERSION_FULL`.** Same build string, different flavor and extension. Bump both jobs together, or the Windows and Linux CI diverge.
- **`MAJOR` is not "the major version".** Despite the name it is whatever directory the CDN uses, which for patch releases is the full number.
- **Verify before editing, not after.** A missing archive discovered after a full sweep means reverting everything.
