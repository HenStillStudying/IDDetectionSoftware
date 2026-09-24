# Deploying to an Ubuntu VPS

Step-by-step runbook for running this stack on a single Ubuntu server with
docker-compose. Written against Ubuntu 22.04/24.04 LTS on **x86_64**, and
driven from a Windows laptop (PowerShell shown where it differs).

**What this gets you:** the full stack (API, worker, OCR, Redis) running,
surviving crashes and reboots, with the API reachable **only through an
SSH tunnel**. That's deliberate, not a missing step — see
[Why no public port yet](#why-no-public-port-yet). Public HTTPS access is
the next step once there's a domain (see [Not covered yet](#not-covered-yet)).

> **Real ID data:** this is still a demo/portfolio-stage system — see the
> README's "Data handling & compliance" section. Don't put other people's
> real KTP photos through a deployed instance until the "before real
> production" items there (consent, per-user auth, retention policy, DPIA)
> are done.

---

## 0. Server sizing

Measured on the actual containers, not guessed:

| | Measured | Recommendation |
|---|---|---|
| RAM | ~0.6 GB idle; ~1.3 GB with 3 requests in flight (OCR ~725 MB, API ~350 MB) — more when the worker is also processing jobs | **2 GB minimum, 4 GB comfortable** |
| Disk | images 4.9 GB, plus build cache while building | **20 GB+ free** |
| CPU | ~1 s of processing per card on a desktop CPU (ONNX engine) | 2+ vCPUs |
| Architecture | x86_64 | **x86_64 only** — ARM (aarch64) is untested; PaddlePaddle wheels may not exist for it at the pinned version |

## 1. First login: a non-root user with SSH keys

On the **laptop**, create a key if you don't already have one:

```powershell
ssh-keygen -t ed25519
```

Log in to the server as root (with whatever the VPS provider gave you) and
create a working user:

```bash
adduser deploy
usermod -aG sudo deploy
```

Copy your public key to it — from the **laptop** (PowerShell has no
`ssh-copy-id`):

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh deploy@YOUR_VPS_IP "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

**Check that key login works** in a *new* terminal before going further:
`ssh deploy@YOUR_VPS_IP` should get you in without a password.

## 2. Lock down SSH

> ⚠️ **Lockout risk.** Keep your current session open until you've
> confirmed a *new* session still gets in. If step 1's key login didn't
> work, stop here.

```bash
# "00-" so it's read first: sshd uses the first value it sees for each
# setting, and some providers ship a 50-cloud-init.conf that sets
# PasswordAuthentication yes — a file named later would lose to it.
sudo tee /etc/ssh/sshd_config.d/00-hardening.conf > /dev/null <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
EOF

sudo sshd -t                                             # syntax check — must print nothing
sudo sshd -T | grep -Ei "^(passwordauthentication|permitrootlogin)"   # must show "no" for both
sudo systemctl restart ssh
```

Then, from a **new** laptop terminal: `ssh deploy@YOUR_VPS_IP` must still
work, and `ssh root@YOUR_VPS_IP` must now be refused.

## 3. Firewall and automatic security updates

```bash
sudo ufw allow OpenSSH          # BEFORE enabling, or you lock yourself out
sudo ufw enable
sudo ufw status                 # should list OpenSSH only

sudo apt-get update
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades   # answer "Yes"
```

> **Docker bypasses ufw.** Any port a container publishes on the public
> interface is reachable from the internet *even though `ufw status` doesn't
> list it* — Docker inserts its own iptables rules ahead of ufw's. That's why
> this repo's `docker-compose.yml` binds the API to `127.0.0.1:8000` only.
> Never change it back to a bare `"8000:8000"` on a server.

## 4. Install Docker

Use Docker's own apt repository (not the Ubuntu `docker.io` package or the
snap). These are the official steps from
<https://docs.docker.com/engine/install/ubuntu/> — check there if anything
fails, since they occasionally change:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl enable --now docker     # start now, and on every boot
sudo usermod -aG docker deploy         # run docker without sudo
```

Log out and back in (the group change needs a new session), then check:
`docker run --rm hello-world` and `docker compose version`.

> Membership of the `docker` group is effectively root access. Only add
> users you'd give sudo to anyway.

## 5. Get the code and the model weights

```bash
git clone https://github.com/HenStillStudying/IDDetectionSoftware.git
cd IDDetectionSoftware
mkdir -p training/runs/ktp_detector/weights
```

The trained detector weights are **not in git** (gitignored as a local
artifact), so copy them from the **laptop**:

```powershell
scp training\runs\ktp_detector\weights\best.pt deploy@YOUR_VPS_IP:~/IDDetectionSoftware/training/runs/ktp_detector/weights/
```

> **If you skip this, nothing errors.** The API and worker silently fall
> back to stub detection (every card "detected" at confidence 0.0 as the
> whole image). Step 8 checks for exactly this.

## 6. Create `.env` on the server

Generate fresh secrets **on the server** — don't copy your laptop's `.env`:

```bash
cat > .env <<EOF
REDIS_PASSWORD=$(openssl rand -hex 32)
KTP_API_KEY=$(openssl rand -hex 32)
KTP_OCR_INTERNAL_KEY=$(openssl rand -hex 32)
EOF
chmod 600 .env
cat .env        # note KTP_API_KEY — clients need it
```

- **Use `-hex`, not `-base64`**: `REDIS_PASSWORD` ends up inside a
  `redis://:password@redis:6379` URL, and base64's `/`, `+` and `=` break
  URL parsing.
- **Set `KTP_API_KEY` on a server.** Blank disables auth entirely — only
  acceptable on a laptop.

## 7. Build and start

```bash
docker compose up -d --build
```

The first build downloads PyTorch and PaddlePaddle and converts the OCR
models — expect several minutes (it took ~14 min on a slow home
connection; a datacenter link is usually much faster). Later rebuilds
after code-only changes take seconds.

Wait until everything is healthy:

```bash
docker compose ps       # api, ocr, redis should show "(healthy)"; worker has no healthcheck
```

## 8. Verify it actually works

Everything below runs through an **SSH tunnel** from the laptop, since the
API isn't public. Open the tunnel in one terminal and leave it running:

```powershell
ssh -N -L 8000:127.0.0.1:8000 deploy@YOUR_VPS_IP
```

In a second terminal (from the repo folder on the laptop), with
`KTP_API_KEY` from step 6:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/v1/ktp/extract -H "X-API-Key: YOUR_KTP_API_KEY" -F "file=@training/sample_ktp.png"
```

Check the extract response for:

- `"status": "ok"` and filled-in fields.
- **`bounding_box.detection_confidence` well above 0** (~0.96 for the
  sample card). **Exactly 0.0 means the weights from step 5 are missing**
  and the stub detector is running.
- `nik_consistency.consistent: false` — expected for `sample_ktp.png`,
  which is a synthetic card whose fields contradict its NIK.

And confirm the negatives:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/ktp/extract -F "file=@training/sample_ktp.png"   # no key -> 401
curl.exe -m 5 http://YOUR_VPS_IP:8000/health                                                 # must FAIL (not public)
```

> **`/demo` doesn't work on a server.** The demo page doesn't send an
> `X-API-Key` header, so with `KTP_API_KEY` set every upload from it gets a
> 401. Use `curl` as above.

## 9. Day-to-day operations

| Task | Command |
|---|---|
| Status | `docker compose ps` |
| Logs (follow) | `docker compose logs -f api` (or `worker`, `ocr`, `redis`) |
| Restart one service | `docker compose restart api` |
| Stop everything | `docker compose down` |
| Deploy a new version | `git pull && docker compose up -d --build` |
| Roll back | `git checkout <previous-commit> && docker compose up -d --build` |
| Free disk (old images/cache) | `docker image prune -f && docker builder prune -f` |

- **Reboots and crashes:** every service has `restart: unless-stopped` and
  Docker starts on boot (step 4), so the stack comes back on its own. Only
  an explicit `docker compose stop`/`down` keeps it down.
- **Logs** rotate at 3 × 10 MB per container, so they can't fill the disk.
- **Backups:** the only state worth keeping is `.env` and the weights file
  — both easy to recreate or re-copy. Redis job data is deliberately
  short-lived (10-minute TTL) and not persisted.

## Why no public port yet

Exposing port 8000 publicly today would mean **plain HTTP**: the
`X-API-Key` header and uploaded KTP photos would cross the internet
unencrypted. The SSH tunnel encrypts both. Public access should arrive
together with HTTPS, not before it.

## Not covered yet

- **HTTPS / public access.** Needs a decision: a **domain** pointed at the
  server (then a reverse proxy such as Caddy gets certificates
  automatically), or IP-only (harder — publicly trusted certificates for
  bare IPs are uncommon). When a proxy is added, it listens on 80/443 (open
  those in ufw) and forwards to `127.0.0.1:8000`.
- **Rate limiting behind a proxy.** The API limits by client IP. Behind a
  reverse proxy, every request arrives from the proxy's own address, so all
  users would share **one** 15/minute bucket — this must be dealt with
  (trusted proxy headers) as part of the HTTPS step, not after.
- **Monitoring/alerting**: nothing watches the server yet beyond Docker's
  own restarts.
- **`/demo` with auth**: the page can't send an API key (see step 8).
