# RPOW2 GPU Miner

Protocol-level RPOW2 miner for NVIDIA GPU servers.

The miner talks directly to `https://api.rpow2.com`, uses CUDA for SHA-256 nonce search, and runs a small API pipeline so the GPU can keep receiving new challenges without waiting for mint submission.

## Files

- `install.sh`: one-command installer; prompts for `rpow_session`.
- `rpow2_gpu.py`: API protocol and pipeline controller.
- `rpow2_cuda_search.cu`: CUDA nonce search helper source.
- `build_cuda.sh`: compiles `/root/rpow2_cuda_search`.
- `rpow2_gpu_fleet.sh`: start/stop/status/log helper.

## Quick Start

Run as `root` on an NVIDIA GPU server:

```bash
git clone YOUR_REPO_URL /root/rpow2-gpu-miner
cd /root/rpow2-gpu-miner
bash install.sh
```

When prompted, paste only the `rpow_session` value, or paste a full cookie string containing `rpow_session=...`.

The installer will:

1. Copy scripts to `/root`.
2. Write `/root/.rpow2_cookie`.
3. Compile `/root/rpow2_cuda_search`.
4. Check `/me` and `/ledger`.
5. Run a small CUDA self-test.
6. Start the default pipeline miner.

## Defaults

Default runtime:

```text
RPOW2_LANES=2
RPOW2_PREFETCH=256
RPOW2_FETCHERS=16
RPOW2_MINTERS=16
```

Override when installing:

```bash
RPOW2_LANES=2 RPOW2_PREFETCH=512 RPOW2_FETCHERS=32 RPOW2_MINTERS=32 bash install.sh
```

If logs show many `SSL EOF`, `Connection reset`, or `HTTP 500`, lower API pressure:

```bash
RPOW2_PREFETCH=128 RPOW2_FETCHERS=8 RPOW2_MINTERS=8 /root/rpow2_gpu_fleet.sh restart 2
```

## Operations

```bash
/root/rpow2_gpu_fleet.sh count
/root/rpow2_gpu_fleet.sh status
/root/rpow2_gpu_fleet.sh balance
/root/rpow2_gpu_fleet.sh tail
/root/rpow2_gpu_fleet.sh stop-all
```

Fast kill if stop is slow:

```bash
pkill -TERM -f 'rpow2_gpu.py run'
pkill -TERM -f 'rpow2_cuda_search'
sleep 2
pkill -KILL -f 'rpow2_gpu.py run'
pkill -KILL -f 'rpow2_cuda_search'
rm -f /root/rpow2_gpu_fleet.pids
```

## Manual Build

```bash
chmod +x /root/build_cuda.sh /root/rpow2_gpu.py /root/rpow2_gpu_fleet.sh
/root/build_cuda.sh
python3 /root/rpow2_gpu.py preflight
/root/rpow2_cuda_search 00 20 --device 0 --progress-ms 1000
/root/rpow2_gpu_fleet.sh restart 2
```

## Publish To Git

From your local machine:

```bash
cd /Users/longmo/Documents/Playground/outputs/rpow2-gpu-miner
git init
git add .
git commit -m "Initial RPOW2 GPU miner"
git branch -M main
git remote add origin YOUR_REPO_URL
git push -u origin main
```
