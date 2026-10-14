# H100 Cluster — Slurm Working Guide

Companion to `H100_AGENT_RULES.md`. That file covers **how to connect** (SSH rules).
This file covers **how to use Slurm** on the cluster. If the two conflict,
`H100_AGENT_RULES.md` wins.

All commands below are run as `ssh -n H100 '<command>'` (see the SSH rules).

---

## 1. Cluster facts (observed 2026-10-07, may change)

| Item | Value |
|---|---|
| Partition | `main` |
| Nodes | `worker-0` … `worker-9` (dynamic nodes) |
| GPUs per node | 8 × `nvidia_h100_80gb_hbm3` |
| CPUs per node | 128 (64 cores × 2 threads) |
| RAM per node | ~2 TB |
| **Fair share per GPU** | **~16 CPUs, ~256 GB RAM** |
| Load | Very high: at last check 78/80 GPUs were in use |

GPUs are the scarce resource. Request only what you need, release as soon as done.

## 2. Check before you request

```bash
# Free GPUs / CPUs / RAM per node (GresUsed shows GPUs in use and their indices)
sinfo -p main -N -O "NodeList:12,GresUsed:45,CPUsState:16,Memory:10,AllocMem:10"

# Your jobs, with time left (%L) and pending reason (%R)
squeue --me -o "%.10i %.20j %.8T %.10M %.10L %.12b %R"

# Dry run: validates the request and estimates start time, submits nothing
sbatch --test-only <same options> --wrap "true"

# Your limits (max GPUs / jobs per user, if configured)
sacctmgr show assoc where user=$USER format=User,Account,Partition,QOS,GrpTRES,MaxTRES
```

## 3. Choose how to run

| Situation | Use | Why |
|---|---|---|
| A task with a clear end (train, eval, benchmark, preprocessing) | `sbatch script.sh` | Releases the GPU automatically when done. **Preferred.** |
| Many similar tasks (sweeps, shards) | `sbatch --array=0-N%K` | `%K` caps how many run at once |
| Task B must wait for task A | `sbatch --dependency=afterok:<A_ID> b.sh` | No manual polling |
| Interactive debugging shared by human + agent | hold job (`sleep infinity`) + `srun --overlap` | Keeps one allocation for both; costs GPU while idle |
| Quick command inside an existing allocation | `srun --overlap --jobid=<ID> ...` | No new allocation |

Do **not** use `salloc` from an agent: it blocks until resources are granted.
Do **not** use `--pty` from an agent: there is no terminal.

## 4. Batch script template

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=<REMOTE_WORKDIR>/logs/%x_%j.out   # %x = job name, %j = job id

set -euo pipefail
source <CONDA_SH> && conda activate <CONDA_ENV>
cd <REMOTE_WORKDIR>

echo "node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
srun python <script>.py <args>
echo "end=$(date -Is)"
```

Rules for scripts:
- Always set `--time`. Estimate, then add ~30% margin. Do not default to many hours.
- Do not set `--nodelist` unless there is a reason (data on a node's local disk,
  team assignment). Pinning a node makes waiting much more likely.
- Make the logs directory first: `mkdir -p <REMOTE_WORKDIR>/logs`.
- Save checkpoints regularly so a job killed by `--time` can resume.

## 5. Hold job (shared interactive allocation)

```bash
sbatch --job-name=<name> --partition=main --nodes=1 --gres=gpu:1 \
  --cpus-per-task=16 --mem=128G --time=04:00:00 \
  --output=/dev/null --wrap "sleep infinity"
```

- `sleep infinity` does nothing; it only keeps the job (and the GPU) alive until
  `--time` ends or `scancel`.
- Human attaches: `srun --overlap --jobid=<ID> --pty bash` (inside tmux on the login node).
- Agent attaches: `srun --overlap --jobid=<ID> <command>` (wrapped in `timeout`).
- Verify the GPU is visible in steps: `srun --overlap --jobid=<ID> nvidia-smi -L`.
  If not, add `--gres=gpu:1` to the `srun` command.
- **Cancel it when the session is over.** An idle hold job blocks others.

## 6. Reading pending reasons (`%R` in squeue)

| Reason | Meaning | Action |
|---|---|---|
| `Resources` | Waiting for resources to free up | Wait, or request less |
| `Priority` | Other jobs are ahead in the queue | Wait |
| `ReqNodeNotAvail` | Requested node(s) cannot run it now (down/drain/reserved, or full when pinned) | Check `sinfo` (section 2); drop `--nodelist` |
| `QOSMax…` / `AssocGrp…` / `AssocMax…` | You hit a per-user or per-account limit | Wait for your other jobs, or ask admin |

Also useful: `squeue --me --start` shows the scheduler's estimated start time.
Full list: https://slurm.schedmd.com/job_reason_codes.html

## 7. Monitoring and results

```bash
tail -n 50 <REMOTE_WORKDIR>/logs/<name>_<id>.out          # progress
sstat -j <ID>.batch --format=JobID,MaxRSS,AveCPU           # live usage of a running batch job
sacct -j <ID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode   # after it ends
srun --overlap --jobid=<ID> nvidia-smi                      # GPU use inside a running job
```

- Poll at most once per 60 s. Prefer checking the log over repeated `squeue`.
- After a job ends, check `State` and `ExitCode` in `sacct` before reporting success.
  `COMPLETED` with exit `0:0` is success; `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`,
  `CANCELLED` are not.

## 8. Permissions for agents

| Action | Allowed |
|---|---|
| `sinfo`, `squeue`, `sacct`, `sstat`, `scontrol show ...`, `sbatch --test-only` | Yes |
| `srun --overlap --jobid=<ID>` inside the allocation named in the rules file | Yes |
| `sbatch` a new job | **Only with human approval.** Show the full script and options first. |
| `scancel` | **Only your own jobs, only with human approval.** Never other users' jobs or steps. |
| `salloc`, `srun --pty`, `scontrol update` | No |

## 9. Etiquette (shared, heavily loaded cluster)

- Request at most the fair share per GPU (~16 CPUs, ~256 GB) unless the workload
  truly needs more.
- Never hold GPUs idle "just in case". Release them, request again later.
- Prefer one well-tested job over many trial-and-error submissions: test on a tiny
  input first with a short `--time` (e.g. 10 minutes).
- Do not run heavy work (training, big installs, large data processing) on the
  login node itself. Use a job.
