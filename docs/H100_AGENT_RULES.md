# H100 Cluster — Rules for Coding Agents

These rules are mandatory. A previous account was banned because an agent opened
~2000 SSH logins. Following these rules keeps all remote work on ONE shared SSH
connection and avoids hung commands.

---

## 0. Config (filled in by the human)

```
SSH_ALIAS      = H100                     # the ONLY alias you may use
JOB_ID         = <JOB_ID>                 # Slurm job you attach to with srun
JOB_OWNER      = <user>                   # if not you: never cancel anything in it
REMOTE_WORKDIR = <path on shared FS>      # your working dir on the cluster
CONDA_SH       = <path>/etc/profile.d/conda.sh
CONDA_ENV      = <env name>
INSTALLER      = <agent name>             # the ONLY agent allowed to install packages
```

## 1. Topology (read-only knowledge)

```
local machine ──(1 shared master connection)──► VSFVPN (jump) ──► login node ──srun──► H100 compute node
```

- The master connection is started and owned by the human. You never start, stop
  or restart it.
- `ssh H100` reuses that master. If the master is down, `ssh H100` silently opens a
  NEW login — which is exactly what got the old account banned.

## 2. SSH rules (hard)

1. Use **only** `ssh -n H100 '...'` (and `scp`/`rsync` to `H100:` for file copy).
   - Never use `h100-master`, an IP address, or any other alias.
   - Never pass `-M`, `-S`, `-F`, `-O exit`, `-o ControlMaster=...`, `-o ControlPath=...`.
2. **Before your first remote command in a task**, and after **any** connection error, run:
   ```bash
   ssh -O check H100
   ```
   This is local only (no login). If the output is not `Master running`, **STOP** and
   ask the human. Do not run any other `ssh` command.
3. **Never retry** after these errors. Stop and report the exact message:
   - `Permission denied`
   - `Connection refused` / `Connection closed` / `Connection timed out`
   - `already exists, disabling multiplexing`
   - `Control socket connect(...)` errors
4. **One SSH call per logical step.** Combine related commands with `&&` or `;` in a
   single call. Do not run one `ssh` per tiny check.
5. **No polling loops faster than 60 s.** Never use `watch`. Never loop `ssh` in a script.
6. Always use `-n` (stdin from /dev/null). Exception: piping a file in
   (`ssh H100 'cat > f' < f`) — prefer `scp` instead.

## 3. Never run interactive commands

No TTY is available; these hang forever:
`vim`, `nano`, `less`, `more`, `top`, `htop`, `watch`, `man`, a bare `python`/`ipython`,
`srun --pty`, `sudo` (password prompt), `git` without `--no-pager`, `conda` prompts
without `-y`.

Set `PAGER=cat` and `GIT_PAGER=cat` in remote commands when in doubt.

## 4. Every remote command needs a timeout and correct exit code

Template:
```bash
ssh -n H100 'timeout 600 bash -c "set -o pipefail; <commands>"'
```

- Do **not** use `cmd | tail -n 3` directly: it hides progress and the exit code.
  Write to a log, then tail the log:
  ```bash
  <cmd> > /tmp/agent_$$.log 2>&1; rc=$?; tail -n 20 /tmp/agent_$$.log; exit $rc
  ```
- Prefer putting multi-step logic in a script instead of deep nested quotes:
  ```bash
  scp task.sh H100:<REMOTE_WORKDIR>/agent_tasks/task.sh
  ssh -n H100 'timeout 900 srun --overlap --jobid=<JOB_ID> bash <REMOTE_WORKDIR>/agent_tasks/task.sh'
  ```

## 5. Slurm rules

- Run GPU work only inside the existing allocation:
  `srun --overlap --jobid=<JOB_ID> ...`, always wrapped in `timeout`.
- Read-only checks are allowed:
  ```bash
  ssh -n H100 'squeue --me; squeue --steps -j <JOB_ID>'
  ```
- **Never** without explicit human approval:
  - `scancel` anything (job or step), especially in jobs you do not own
  - `sbatch` / `salloc` / a new `srun` allocation (costs GPU time)
- If `srun` seems stuck, check `squeue --steps -j <JOB_ID>` once and report. Do not
  launch more `srun` steps to "try again".

### 5b. The GPU is shared with the human

The allocation `<JOB_ID>` has ONE GPU, used by both the human (interactive shell)
and you. Running GPU work at the same time can cause CUDA out-of-memory for both.

- Before starting any GPU work, check memory use in one call:
  ```bash
  ssh -n H100 'timeout 60 srun --overlap --jobid=<JOB_ID> nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv'
  ```
  If another process is already using a large share of GPU memory, ask the human
  before starting.
- Check the time left before long work (`%L` = time left):
  ```bash
  ssh -n H100 'squeue -j <JOB_ID> -o "%.10i %.10L"'
  ```
  Do not start work that cannot finish before the allocation ends.
- Never request a new allocation if this one ends; tell the human.

## 6. Long-running work (> 10 min)

Do not hold an SSH session open. Start it detached on the login node, write to a log,
return immediately, then check the log occasionally (>= 60 s apart).

```bash
# start (check that tmux exists first: command -v tmux)
ssh -n H100 'tmux new -d -s <task_name> "srun --overlap --jobid=<JOB_ID> bash <script> > <REMOTE_WORKDIR>/logs/<task_name>.log 2>&1"'

# check
ssh -n H100 'tail -n 30 <REMOTE_WORKDIR>/logs/<task_name>.log; tmux has-session -t <task_name> && echo RUNNING || echo DONE'
```

## 7. Environment and installs

- Only `<INSTALLER>` may run `pip install`, `uv pip install`, `conda install`.
  Other agents: ask the installer or the human.
- Never run two installs into the same env at the same time.
- Activate env with:
  `source <CONDA_SH> && conda activate <CONDA_ENV>`

## 8. Filesystem

- Write only inside `<REMOTE_WORKDIR>` and `/tmp`.
- Never `rm -rf` outside `<REMOTE_WORKDIR>`. Never touch other users' directories.
- `/mnt/data` is a shared network filesystem: avoid many small file operations in loops.

## 9. When in doubt

Stop and ask the human. A paused task costs minutes; a banned account costs days.
