# VfxDB Downloader Specification

This document is the normative product contract for the VfxDB downloader. It
describes what a user asks for, which local files are authoritative, how whole
archives are selected, and what must be true before a download is considered
successful.

The words **must**, **must not**, **should**, and **may** are requirements in
this document. The implementation must not preserve an older behavior when it
conflicts with this specification.

## 1. Product model

The downloader has two deliberately separate jobs:

1. Install the complete local JSON control data needed to understand VfxDB.
2. Optionally download VDB data using one of three data-selection modes.

Every valid invocation must finish the first job before planning or downloading
any VDB archive. JSON control data is mandatory; it is not an optional add-on
and there is no `--metadata` switch.

The only user-facing data unit is a complete tar archive. The downloader must
not select individual VDB members from a tar, must not download a tar in order
to inspect which sequence it contains, and must not search every remote tar for
a sequence. A local category index maps a sequence directly to its tar.

## 2. Mandatory local JSON control data

For a valid invocation, the downloader must first pin one immutable Hugging
Face dataset revision and install all of the following from that same revision:

- Every top-level `<Category>/category_index.json` in the dataset repository.
- `meta/vfxdb_meta.tar.zst`, fully unpacked so that every published
  `<Category>/index/*.json` is present locally, except JSON files deliberately
  removed by the IO-bad rule in section 8.
- `meta/io_bad_vdbs.jsonl` and `meta/io_bad_vdbs.meta.json`.

Category names are discovered from repository paths that exactly match
`<Category>/category_index.json`. `dataset_index.json` is not an input to
discovery, planning, selection, or installation.

All category indexes must be downloaded, including categories not named in a
later category-specific request. Each downloaded `category_index.json` must
retain every upstream row and every upstream field. The only permitted local
addition is `deleted_bad_io_sample` on an IO-bad sample row, as defined in
section 8. The downloader must not generate a replacement category index,
reduce its sample list, recompute its counts, or remove fields.

After these files are local, all data planning must read the local
`<Category>/category_index.json` files and local IO-bad list. A remote index,
`dataset_index.json`, remote archive scan, or previously generated private
index must not silently replace them.

For each sample row, the first component of `vdb_path` identifies its sequence
and therefore its archive directly:

```text
<Category>/<sequence>/<file>.vdb
        -> archives/<Category>/<sequence>.tar
```

Distinct sequence components are distinct tar units. Within a category, the
fixed tar order is the order in which each distinct sequence first appears in
that category's local `category_index.json`. Repeated runs against the same
revision therefore produce the same selection. Categories are processed in
lexicographic category-name order when an algorithm needs a stable order among
categories.

## 3. Command-line interface

The public command is:

```bash
python tools/download_extract_data.py [DATA_ROOT] [OPTIONS]
```

`DATA_ROOT` defaults to `data/vdbset`.

There are exactly three mutually exclusive data-selection modes:

### 3.1 Preset mode

```bash
python tools/download_extract_data.py /data/vfxdb --preset smoke
python tools/download_extract_data.py /data/vfxdb --preset medium
python tools/download_extract_data.py /data/vfxdb --preset full
```

`--preset` must not be combined with `--percentage`, `--category`, or
`--max-samples`.

### 3.2 All-category percentage mode

```bash
python tools/download_extract_data.py /data/vfxdb --percentage 10
```

One percentage applies to the complete set of categories. Percentage mode does
not accept `--category` or `--max-samples`. The percentage must be greater than
zero and at most 100.

### 3.3 Category and maximum-sample mode

```bash
python tools/download_extract_data.py /data/vfxdb \
  --category CloudWave \
  --max-samples 1000

python tools/download_extract_data.py /data/vfxdb \
  --category CloudWave \
  --category SurfaceFire \
  --max-samples 1000
```

`--category` may be repeated. The same positive `--max-samples` value applies
separately to every requested category. This mode requires both arguments; a
category without a maximum, or a maximum without a category, is an error.

### 3.4 Interactive terminal interface

```bash
python tools/download_extract_data.py /data/vfxdb --tui
```

`--tui` is an interface, not another selection mode. It must reject
`--preset`, `--percentage`, `--category`, and `--max-samples`; destination,
revision, cache, and the initial IO-bad preference may be provided as defaults.
The ordinary bare command must never enter the TUI automatically.

The interactive order is normative:

1. Confirm the destination and whether to start mandatory-file preparation.
   Revision, cache, and the explicit diagnostic IO-bad override are grouped
   under advanced settings; the ordinary path does not require users to learn
   the internal bad-sample filtering mechanism.
2. Prepare and validate every category index, both named IO-bad control files,
   and every indexed `<Category>/index/*.json` exactly as the plain CLI does.
3. Present all six choices using the installed local category indexes: required
   JSON only, Smoke, Medium, Full, global percentage, or selected categories
   with one shared maximum sample count.
4. Generate the exact `DownloadPlan` from those local indexes.
5. Show revision, destination, cache, exact tar count, usable sample counts,
   per-category allocation, whole-tar target overshoot, conservative network
   and installed-volume upper bounds, and free space on the destination and
   cache filesystems. An explicit IO-bad override must be shown as a warning;
   default filtering remains transparent.
6. Require an explicit three-way decision before the first VDB tar download:
   download, return to selection, or quit. Returning to selection must not
   repeat mandatory JSON preparation. Full mode requires the exact text
   `FULL`; invalid input is retried rather than treated as an implicit quit.

Rejecting the preparation prompt performs no Hub access and no destination
write. Quitting from selection or the exact VDB plan returns success, keeps all
mandatory JSON, and downloads no VDB tar. `Ctrl-C` returns 130 after restoring the terminal;
completed cache objects and installed tars remain reusable. Other errors return
1 and must identify the stage, current archive/path, concrete cause, and rerun
action without printing a traceback.

The Rich view must not change downloader semantics. It uses core events and
Hugging Face's own byte progress where supported; older Hub versions suppress
their fallback bar only for the active call and use a stage indicator without
replacing the HF transfer implementation. It must not leave Hub progress
globally disabled after returning. Overall tar progress, current-file bytes,
and per-sample JSON work are distinct units and must never overwrite one
another. Non-TTY `--tui` exits 2 before Hub or filesystem access. Below 60
columns the view reduces to single-column numeric status; `NO_COLOR` and narrow
terminals retain the same labels and decisions without depending on color or
Unicode for meaning. A TTY reporting `TERM=dumb` uses periodic plain progress
lines instead of cursor-addressed Live rendering.

The following infrastructure options may be combined with any valid mode or
with a bare invocation:

- `--include-bad`: retain selected IO-bad files after extraction.
- `--revision REVISION`: select the Hugging Face revision to pin.
- `--cache-dir PATH`: select the Hugging Face Hub cache location. Complete Hub
  objects and completed local tar installations are reused on rerun.

These are not additional data-selection modes. In particular,
`--include-bad` must not change archive selection, requested quotas, ordering,
or normal-sample counts.

Old selection controls such as `--all`, `--limit`, and `--metadata` are not
part of this interface and must be rejected rather than assigned legacy
semantics.

## 4. Bare invocation

A command with no data-selection option is valid:

```bash
python tools/download_extract_data.py /data/vfxdb
```

It must:

1. Install and validate every `category_index.json`.
2. Download and fully unpack `meta/vfxdb_meta.tar.zst`.
3. Install and validate the IO-bad list and its integrity JSON.
4. Apply the section 8 IO-bad rule to the installed JSON files and category
   index annotations.
5. Download no `archives/<Category>/<sequence>.tar` files.
6. Clearly state that no VDB data was downloaded.
7. Print concise examples for Smoke, Medium, Full, all-category percentage, and
   category plus maximum-sample modes.

This is the supported way to prepare or refresh the local control data without
downloading VDB data. It must not silently choose a starter dataset.

## 5. Presets

### 5.1 Smoke

Smoke selects the first 2 tar units from every category, using each category's
fixed tar order.

- A category with fewer than 2 tars contributes all of its tars.
- Its shortfall is not redistributed to other categories.
- Selection and download remain whole-tar operations.

Formally, for category `c` with ordered tar list `T_c`, Smoke selects
`T_c[:2]`.

### 5.2 Medium

Let `A` be the number of distinct tar units across all categories. Medium's
global target is:

```text
ceil(A * 0.20)
```

Tars are assigned by balanced rounds:

1. In a round, take the next tar from every category that still has one.
2. Stop immediately when the global target is reached.
3. Once a category is exhausted, skip it in later rounds.
4. Continue filling from categories with remaining tars until the exact target
   number of tars has been selected.

Thus categories receive the same number of tars while possible, and categories
with more tars supply the remainder only after shorter categories are
exhausted. The selected tar count is exactly the target; archive boundaries do
not require further rounding because the target is already a tar count.

### 5.3 Full

Full selects every tar unit in every category.

## 6. All-category percentage mode

For requested percentage `P`, let `A` be the total number of distinct tar units
across all categories. The global target is:

```text
ceil(A * P / 100)
```

Selection uses the same balanced-round algorithm as Medium. A single `P`
always applies to all categories together; per-category percentages are not
supported. A request for 100 percent is exactly equivalent to Full, including
the deterministic category-major download order.

The console summary must report the requested percentage, total available tar
count, selected tar count, and per-category selected tar count before data
download begins.

## 7. Category and maximum-sample mode

Every row of a local `category_index.json` represents one sample. Samples known
to be IO-bad do not count toward `--max-samples`, regardless of whether
`--include-bad` is present.

For each requested category independently:

1. Walk complete tars in that category's fixed order.
2. Add the number of non-IO-bad sample rows belonging to each tar.
3. Select the complete tar and continue until the cumulative normal-sample
   count is greater than or equal to `--max-samples`.
4. If the requested maximum is greater than the category's total normal-sample
   count, select the category in full.

The result deliberately rounds upward at the tar boundary. For example, if
selected tars contain 900 normal samples and the next complete tar raises the
count to 1,100, a request for 1,000 selects that next tar and yields 1,100
normal samples. No tar is partially extracted to enforce the numeric maximum.

Tars containing zero normal samples may occur. Walking the fixed order still
selects such a tar before continuing; its zero contribution must not terminate
the loop or change the order.

### 7.1 EnvironmentalFog

`EnvironmentalFog` is single-frame data. Each VDB row in its local
`category_index.json` counts as one independent sample; the downloader must not
invent a previous frame, temporal relationship, or synthetic sequence. Its
storage and download unit remains the complete tar, exactly like every other
category:

- Smoke counts its tar units, not individual frames.
- Medium and percentage mode include its tar units in balanced allocation.
- Category maximum mode accumulates its independent non-IO-bad VDB rows per
  complete tar and rounds upward at the tar boundary.

## 8. IO-bad behavior

The IO-bad list is a mandatory lower-level rule, not a fourth selection mode.
Normal user-visible quotas and sample counts exclude IO-bad rows. The archive
plan must be identical with and without `--include-bad`; the flag controls only
whether bad files found inside already-selected tars are retained.

The cleanup occurs after complete downloads and extraction:

### 8.1 Default behavior

For every row identified by `meta/io_bad_vdbs.jsonl`, the downloader must:

1. Remove the installed VDB if it exists.
2. Remove the row's `meta_path` file under `<Category>/index/` if it exists.
3. Preserve the row and all of its original fields in
   `<Category>/category_index.json`.
4. Add or set:

   ```json
   "deleted_bad_io_sample": true
   ```

The deletion and annotation are idempotent. A missing bad file is already in
the desired state and is not an error. A successfully finalized default data
root exposes neither the bad VDB nor its corresponding per-sample JSON.

### 8.2 Explicit `--include-bad`

For the same IO-bad rows, the downloader must retain or restore files from the
mandatory JSON archive and from any selected VDB tars, while preserving the
same tar plan and normal-sample counts. It must add or set:

```json
"deleted_bad_io_sample": false
```

The annotation is required even when bad files are explicitly retained. It is
the only behavior change in a category index between the two policies. A bad
row whose tar was not selected remains in the category index with `false`, but
no unselected VDB is downloaded merely because `--include-bad` is present.

Switching policies on a later run must converge to the newly requested state:
default mode removes previously retained bad files and changes annotations to
`true`; `--include-bad` restores applicable JSON files, retains bad members of
newly extracted selected tars, and changes annotations to `false`.

Archive validation manifests are internal validation inputs. They must not be
installed as an alternative user-facing index that makes a deleted bad sample
appear available.

## 9. Whole-tar download and extraction

For every selected tar, the downloader must:

1. Construct its exact remote path directly from the category and sequence
   learned from the local category index.
2. Ask `huggingface_hub` to download that one complete file at the pinned
   revision.
3. Reuse the standard Hugging Face cache. The normal script path uses bounded
   HTTP reads and retries so a stalled transfer returns control to the retry
   loop instead of holding the destination lock forever.
4. Validate the complete tar and its published sequence manifest before making
   extracted VDB files visible in the final destination.
5. Extract every VDB member in the tar as a unit; member-level selection is
   forbidden.
6. Apply the IO-bad cleanup policy only after the complete tar is installed.

The complete tar object must always match its published Hugging Face content
identity before extraction. Legacy sequence manifests across multiple
categories contain empty or stale per-member `sha256` values, so those fields
are advisory rather than an integrity root. The downloader instead requires
the verified outer tar, exact manifest membership, canonical paths, and exact
agreement between indexed and tar-member byte sizes. A nonempty member digest
must still be syntactically valid.

If several requested samples map to the same tar, the tar is planned,
downloaded, verified, and extracted once. Existing verified tars and files are
reused on rerun. An interrupted run must be restartable without redownloading a
valid complete Hugging Face object or treating a partial extraction as
complete. The currently transferring tar may restart from zero when the
installed `huggingface_hub` version does not preserve partial objects across
processes; every earlier completed tar remains reusable.

The metadata archive is also downloaded through `huggingface_hub` and its
cache. Safe extraction must reject absolute paths, `..` traversal, links,
devices, duplicate conflicting members, malformed JSON, and content that does
not match published integrity information.

## 10. Local layout

After a normal download, the relevant public layout is:

```text
DATA_ROOT/
├── <Category>/
│   ├── category_index.json
│   ├── index/
│   │   └── <sample>.json
│   └── <sequence>/
│       └── <sample>.vdb
└── meta/
    ├── io_bad_vdbs.jsonl
    └── io_bad_vdbs.meta.json
```

Every category index is the complete upstream index with only the IO-bad
annotation added. It may therefore reference VDBs not selected by the current
preset or quota; downstream training already determines local availability by
checking whether the referenced VDB and per-sample JSON exist. The downloader
must not crop the index to hide unselected normal data.

### 10.1 Training use of `<Category>/index/*.json`

The training loader must use each category-index row's exact `meta_path`; it
must not derive a JSON filename from the VDB filename. When `return_meta=True`,
`build_dataset_splits()` implicitly requires that JSON, and both VDB and NPZ
samples return its decoded top-level object as `sample["meta"]`. A missing,
unreadable, malformed, non-object, or invalid-bbox JSON is a hard error. It
must not be converted to `None` or hidden by advancing to a different sample.
Because category JSON objects may have different keys, batch collation keeps
them as `batch["meta"]: list[dict]` while collating tensor fields normally.

For temporal sampling, explicit `seq_bbox_*` / `sequence_bbox_*` fields are
authoritative. When the JSON files provide only per-frame `bbox_min` and
`bbox_max`, the loader must use the union across every installed frame of that
sequence; treating the first frame's box as the sequence box is forbidden.

## 11. Status and console behavior

Before downloading data tars, the console must make the request legible without
requiring knowledge of repository internals. It must show at least:

- Pinned revision.
- Destination and Hugging Face cache location.
- Selected mode and its value.
- Total and selected tar counts, globally and by category.
- Normal-sample counts for category maximum mode.
- Estimated remote bytes when the repository exposes sizes.
- Whether selected IO-bad files will be removed or retained.

Progress must distinguish mandatory JSON preparation, tar download, tar
verification, extraction, IO-bad cleanup, and completion. A cache hit must be
reported as reuse, not as a fresh network download. The final message must
report actual installed tar and normal-sample counts and explicitly mention any
upward rounding caused by a complete tar.

A bare invocation must end successfully after mandatory JSON preparation and
must prominently say that it downloaded no VDB data. It must print examples of
all three data-selection modes rather than silently doing nothing.

## 12. State, cache, and reruns

- All files for a destination must come from one pinned immutable Hugging Face
  commit. A branch or tag supplied by the user is resolved before installation.
- The implementation may keep minimal internal state needed for revision
  pinning, locking, atomic installation, and resumption. That state must not
  become another source of dataset membership.
- Hugging Face's standard cache is authoritative for downloaded blobs. The
  implementation should not build a competing content cache.
- A second identical run must verify and reuse complete local work.
- A nonempty destination without this downloader's revision state must be
  rejected. Silently adopting files from an old script or manual copy could
  mix dataset revisions; migration, if added later, must be explicit and
  verified.
- `Ctrl-C`, a process crash, or a transient network failure must leave a state
  from which rerunning the same command safely continues.
- Concurrent writers to the same destination must be prevented by a lock;
  readers must never observe a partially committed tar extraction or partially
  rewritten category index.
- Publishing category indexes and IO-bad controls must use a durable
  transaction. A failure rolls every already-published control back, and a
  later invocation recovers a transaction interrupted by process death.
- Cache and extraction space requirements on the same filesystem must be
  aggregated before data transfer. Space reserved for control-file backups and
  atomic publication must also be checked before publication begins.
- IO-bad cleanup and index annotation must be repeatable and must reconcile a
  change between default and `--include-bad` policy.
- Because one IO-bad policy transition spans many files and category indexes,
  the downloader must keep a durable internal transition marker until both
  controls and files reach the requested policy. The production loader must
  refuse to enumerate the data root while that marker exists. A failed or
  interrupted transition is completed by rerunning the downloader; it must
  never become a window in which training silently consumes bad samples.

## 13. Failure semantics

Command-line mode conflicts and invalid values must fail before network or
filesystem work. For a syntactically valid request, mandatory JSON preparation
must complete before any data tar begins downloading.

The downloader must exit nonzero with a concrete path and cause when any of the
following occurs:

- A category index is missing, malformed, internally inconsistent, or contains
  an unsafe path.
- The IO-bad list fails its integrity JSON.
- `meta/vfxdb_meta.tar.zst` is missing, corrupt, unsafe, or incomplete.
- Any sample row lacks a usable `meta_path`, or its referenced JSON is absent
  from the full metadata archive. This includes IO-bad rows because
  `--include-bad` must be able to restore their published JSON before data-tar
  selection begins.
- A requested category does not exist locally.
- A selected tar is missing, corrupt, unsafe, or does not match its sequence
  manifest.
- Available disk space or inodes are insufficient for a safe installation.
- The destination is already pinned to an incompatible revision and cannot be
  reconciled without mixing revisions.
- IO-bad cleanup or atomic category-index annotation cannot finish.

No later data tar may be attempted after a mandatory-control failure. A failed
tar must not leave any of its VDB members marked complete. Cached complete
downloads and diagnostic artifacts should be retained so a rerun can resume or
so the failure can be investigated.

## 14. Required tests and acceptance criteria

Tests must be deterministic, must not depend on the live Hugging Face dataset,
and must exercise the public CLI as well as selection helpers. A small fixture
repository must include multiple uneven categories, a short category, IO-bad
rows, a bad-only tar, EnvironmentalFog frames, valid archive manifests, and a
full metadata archive.

### 14.1 CLI contract

- Bare invocation installs all mandatory JSON and requests zero data tars.
- Bare output names Smoke, Medium, Full, percentage, and category maximum usage.
- Each valid mode parses with infrastructure options.
- Preset plus percentage/category/maximum is rejected.
- Percentage plus category/maximum is rejected.
- Category without maximum and maximum without category are rejected.
- Zero, negative, and over-100 percentages are rejected.
- Nonpositive maximum-sample values and unknown categories are rejected.
- Removed legacy selectors are rejected.

### 14.2 Mandatory JSON ordering

- All category indexes, all published `<Category>/index/*.json` files, and both
  IO-bad files are local and validated before the first data-tar request.
- All category indexes are installed even for category-specific mode.
- Planning reads local category indexes after installation.
- No selection code reads `dataset_index.json`.
- Missing referenced JSON for a normal row fails before any data-tar request.

### 14.3 Fixed ordering and presets

- Tar order follows sequence first appearance in each local category index,
  including nonnumeric sequence names.
- Smoke selects 2 tars per category, selects a short category in full, and does
  not redistribute its shortfall.
- Medium selects exactly `ceil(total_tars * 0.20)`.
- Medium gives categories equal tar counts while possible and then fills only
  from categories with remaining tars.
- Full selects every tar exactly once.
- Repeated planning against identical inputs yields byte-for-byte identical
  plans.

### 14.4 Percentage selection

- Several percentages, including fractional percentages and 100, select
  exactly `ceil(total_tars * P / 100)` tars.
- Uneven categories are balanced round by round and exhausted categories are
  skipped.
- Percentage cannot be scoped to named categories.

### 14.5 Category maximum and EnvironmentalFog

- The same maximum is applied independently to every repeated category.
- The last complete tar is retained when it pushes the sample count above the
  maximum.
- A maximum above a category's total selects the category in full.
- IO-bad rows do not contribute to the cumulative normal-sample count.
- A zero-normal-sample tar does not break accumulation or reorder later tars.
- Every EnvironmentalFog VDB row counts once; no temporal grouping is invented.
- EnvironmentalFog still downloads and extracts whole tars.

### 14.6 IO-bad behavior

- Default and `--include-bad` produce exactly the same tar plan and normal
  quota counts.
- Default removes both the bad VDB and its referenced per-sample JSON and sets
  `deleted_bad_io_sample` to `true`.
- `--include-bad` retains applicable files and sets the same key to `false`.
- All original index rows, row order, top-level fields, and row fields remain
  unchanged apart from that one annotation.
- Cleanup is idempotent when files are already absent.
- Rerunning in the opposite policy reconciles files and annotations.
- Unselected bad VDBs are not fetched merely because `--include-bad` is set.

### 14.7 Archive, cache, resume, and safety

- Every selected tar is requested once and every member is extracted before
  IO-bad cleanup; no selected-member extraction path exists.
- Archive paths are derived directly from local indexes; the implementation
  never scans all remote tars looking for sequences.
- A cache hit avoids a fresh network transfer.
- After an interrupted run, every complete cached tar and every complete local
  installation is reused; an incomplete current transfer may restart according
  to the installed Hugging Face Hub version. Partial extraction is never
  exposed as complete.
- A corrupted cached object is detected and recovered through a fresh Hugging
  Face download when possible.
- Unsafe tar members, manifest mismatches, truncated archives, disk-full
  failures, and concurrent destination writers fail without exposing partial
  successful state.

### 14.8 Release gate

The implementation is acceptable only when:

1. The complete automated downloader test suite passes.
2. A clean fixture run passes for bare, Smoke, Medium, Full, percentage, and
   repeated-category maximum modes.
3. Default and `--include-bad` filesystem trees match the expected policy.
4. An interruption/rerun integration test proves completed-tar cache reuse and
   atomic local installation.
5. The production training loader can enumerate the installed normal samples
   from the untouched category-index rows and installed per-sample JSON files.

Any validation-only runtime instrumentation must remain outside production
training and inference code.
