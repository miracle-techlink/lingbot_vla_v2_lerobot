#!/usr/bin/env python3
"""Minimal paramiko wrapper for the liuyue A100 box (password auth, no key install).

Usage:
  a100.py exec "command string"        # run remote command, stream stdout/stderr
  a100.py put <local> <remote>         # sftp upload (file or dir, recursive)
  a100.py get <remote> <local>         # sftp download (file or dir, recursive)
"""
import os
import stat
import sys

import paramiko

HOST, PORT, USER = "8.130.97.174", 1014, "liuyue"
PASS = open(os.path.expanduser("~/.cache/.a100_pass")).read().strip()


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS)
    return c


def do_exec(cmd):
    c = connect()
    _, out, err = c.exec_command(cmd, timeout=None)
    while True:
        line = out.readline()
        if not line:
            break
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = out.channel.recv_exit_status()
    e = err.read().decode()
    if e.strip():
        sys.stderr.write(e)
    c.close()
    sys.exit(rc)


def _put_recursive(sftp, local, remote):
    if os.path.isfile(local):
        sftp.put(local, remote)
        return
    try:
        sftp.mkdir(remote)
    except OSError:
        pass
    for name in os.listdir(local):
        _put_recursive(sftp, os.path.join(local, name), remote + "/" + name)


def _get_recursive(sftp, remote, local):
    st = sftp.stat(remote)
    if not stat.S_ISDIR(st.st_mode):
        sftp.get(remote, local)
        return
    os.makedirs(local, exist_ok=True)
    for name in sftp.listdir(remote):
        _get_recursive(sftp, remote + "/" + name, os.path.join(local, name))


def main():
    mode = sys.argv[1]
    c = connect()
    if mode == "put":
        sftp = c.open_sftp()
        _put_recursive(sftp, sys.argv[2], sys.argv[3])
    elif mode == "get":
        sftp = c.open_sftp()
        _get_recursive(sftp, sys.argv[2], sys.argv[3])
    c.close()


if __name__ == "__main__":
    if sys.argv[1] == "exec":
        do_exec(sys.argv[2])
    main()
