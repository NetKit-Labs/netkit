#!/usr/bin/env bash
# expect-based ssh/scp helper (no sshpass). Password via NETKIT_PI_PASS.
set -euo pipefail

HOST="${NETKIT_PI_HOST:-192.168.0.176}"
USER="${NETKIT_PI_USER:-pi}"
PASS="${NETKIT_PI_PASS:?set NETKIT_PI_PASS}"
export NETKIT_PI_PASS NETKIT_PI_HOST="$HOST" NETKIT_PI_USER="$USER"

cmd="${1:?ssh|scp|rsync_to}"; shift

case "$cmd" in
  ssh)
    remote_cmd="${1:?remote command}"
    export NETKIT_PI_REMOTE_CMD="$remote_cmd"
    expect <<'EXPECT_EOF'
set timeout -1
log_user 1
set pass $env(NETKIT_PI_PASS)
set rcmd $env(NETKIT_PI_REMOTE_CMD)
set user $env(NETKIT_PI_USER)
set host $env(NETKIT_PI_HOST)
# ssh joins remote argv with spaces and drops quoting, so pass ONE remote
# string: bash -lc '<shell-quoted rcmd>'.
proc sh_single_quote {s} {
  return '[string map {' '{'\''}'} $s]'
}
set remote "bash -lc [sh_single_quote $rcmd]"
set argv_list [list ssh -o StrictHostKeyChecking=accept-new $user@$host $remote]
spawn {*}$argv_list
expect {
  -re "(?i)are you sure you want to continue connecting" {
    send "yes\r"
    exp_continue
  }
  -re "(?i)password:" {
    send -- "$pass\r"
  }
  eof
}
expect {
  -re "(?i)password:" {
    send -- "$pass\r"
    exp_continue
  }
  eof
}
catch wait result
exit [lindex $result 3]
EXPECT_EOF
    ;;
  scp)
    src="${1:?src}"; dst="${2:?dst}"
    export NETKIT_PI_SCP_SRC="$src" NETKIT_PI_SCP_DST="$dst"
    expect <<'EXPECT_EOF'
set timeout -1
log_user 1
set pass $env(NETKIT_PI_PASS)
set argv_list [list scp -o StrictHostKeyChecking=accept-new -r \
  $env(NETKIT_PI_SCP_SRC) \
  $env(NETKIT_PI_USER)@$env(NETKIT_PI_HOST):$env(NETKIT_PI_SCP_DST)]
spawn {*}$argv_list
expect {
  -re "(?i)are you sure you want to continue connecting" {
    send "yes\r"
    exp_continue
  }
  -re "(?i)password:" {
    send -- "$pass\r"
  }
  eof
}
expect {
  -re "(?i)password:" {
    send -- "$pass\r"
    exp_continue
  }
  eof
}
catch wait result
exit [lindex $result 3]
EXPECT_EOF
    ;;
  rsync_to)
    src="${1:?src}"; dst="${2:?dst}"
    export NETKIT_PI_RSYNC_SRC="$src" NETKIT_PI_RSYNC_DST="$dst"
    if command -v rsync >/dev/null 2>&1; then
      expect <<'EXPECT_EOF'
set timeout -1
log_user 1
set pass $env(NETKIT_PI_PASS)
set argv_list [list rsync -az --delete --exclude .venv --exclude .netkit_pi_pass -e {ssh -o StrictHostKeyChecking=accept-new} \
  $env(NETKIT_PI_RSYNC_SRC)/ \
  $env(NETKIT_PI_USER)@$env(NETKIT_PI_HOST):$env(NETKIT_PI_RSYNC_DST)/]
spawn {*}$argv_list
expect {
  -re "(?i)are you sure you want to continue connecting" {
    send "yes\r"
    exp_continue
  }
  -re "(?i)password:" {
    send -- "$pass\r"
  }
  eof
}
expect {
  -re "(?i)password:" {
    send -- "$pass\r"
    exp_continue
  }
  eof
}
catch wait result
exit [lindex $result 3]
EXPECT_EOF
    else
      "$0" scp "$src" "$dst"
    fi
    ;;
  *)
    echo "usage: $0 ssh|scp|rsync_to ..." >&2
    exit 2
    ;;
esac
