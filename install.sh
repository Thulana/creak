#!/bin/sh
# Installs or upgrades creak from GitHub Releases, then runs `creak setup`.
#   curl -fsSL https://raw.githubusercontent.com/Thulana/creak/main/install.sh | sh
# Environment:
#   CREAK_VERSION=v0.1.0   install that release instead of the latest
#   CREAK_NO_SETUP=1       only install the binary
set -eu

REPO="Thulana/creak"
BIN_DIR="/usr/local/bin"

say() { printf 'creak: %s\n' "$*"; }
die() { say "$*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || die "only Linux is supported for now"
case "$(uname -m)" in
  x86_64 | amd64) target="x86_64-unknown-linux-gnu" ;;
  aarch64 | arm64) target="aarch64-unknown-linux-gnu" ;;
  *) die "unsupported architecture $(uname -m)" ;;
esac

if [ "$(id -u)" -eq 0 ]; then
  sudo=""
elif command -v sudo >/dev/null 2>&1; then
  sudo="sudo"
else
  die "run this as root, or install sudo"
fi

if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -qO "$2" "$1"; }
else
  die "needs curl or wget"
fi

version="${CREAK_VERSION:-}"
if [ -z "$version" ]; then
  tmp_latest="$(mktemp)"
  fetch "https://api.github.com/repos/$REPO/releases/latest" "$tmp_latest" || die "could not look up the latest release"
  version="$(sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' "$tmp_latest" | head -n 1)"
  rm -f "$tmp_latest"
  [ -n "$version" ] || die "could not find the latest release"
fi

name="creak-$version-$target"
dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT
base="https://github.com/$REPO/releases/download/$version"

say "downloading $name"
fetch "$base/$name.tar.gz" "$dir/$name.tar.gz" || die "download failed: $base/$name.tar.gz"
fetch "$base/SHA256SUMS" "$dir/SHA256SUMS" || die "download failed: $base/SHA256SUMS"

expected="$(grep " $name.tar.gz\$" "$dir/SHA256SUMS" | cut -d ' ' -f 1)"
[ -n "$expected" ] || die "$name.tar.gz is not listed in SHA256SUMS"
actual="$(sha256sum "$dir/$name.tar.gz" | cut -d ' ' -f 1)"
[ "$expected" = "$actual" ] || die "checksum mismatch for $name.tar.gz; not installing"

if ! ldconfig -p 2>/dev/null | grep -q 'libasound\.so\.2'; then
  command -v apt-get >/dev/null 2>&1 || die "install the ALSA library (libasound2) first"
  say "installing the ALSA library"
  $sudo apt-get update -qq
  # Ubuntu 24.04 renamed the package for the 64-bit time_t transition.
  $sudo apt-get install -y -qq libasound2t64 >/dev/null 2>&1 || $sudo apt-get install -y -qq libasound2 >/dev/null
fi

tar -xzf "$dir/$name.tar.gz" -C "$dir"
$sudo install -m 0755 "$dir/$name/creak" "$BIN_DIR/creak"
say "installed $BIN_DIR/creak ($version)"

if [ "${CREAK_NO_SETUP:-0}" = "1" ]; then
  say "skipping setup; run 'sudo creak setup' to install the service"
else
  $sudo "$BIN_DIR/creak" setup
fi
