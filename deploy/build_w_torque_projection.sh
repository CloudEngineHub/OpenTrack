#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf build/
mkdir build
cd build/
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_DANCE_TORQUE_PROJECTION=ON
make -j20
cd ..

# Ensure SONAME links exist for runtime loader.
ARCH=$(uname -m)
DDS_LIB_DIR="thirdparty/lib/${ARCH}"
if [ -d "$DDS_LIB_DIR" ]; then
	[ -f "$DDS_LIB_DIR/libddsc.so.0" ] || ln -sf libddsc.so "$DDS_LIB_DIR/libddsc.so.0"
	[ -f "$DDS_LIB_DIR/libddscxx.so.0" ] || ln -sf libddscxx.so "$DDS_LIB_DIR/libddscxx.so.0"
fi
