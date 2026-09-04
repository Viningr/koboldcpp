# KoboldCpp Snapdragon backends on the Fold 8 Ultra

This branch is an experimental Android/Termux port of the current llama.cpp OpenCL and Hexagon backends into KoboldCpp. It provides one backend registry containing CPU, Qualcomm Adreno OpenCL, and Qualcomm Hexagon HTP devices.

## Pinned inputs

- KoboldCpp base: `4ac5721b5eb7c5f51216a7839929318b65fd9951` (`LostRuins/koboldcpp`, branch base for this work)
- Imported llama.cpp backend source: `0f3a71be15af836d277c9f918adfafb45732677e`
- Builder image: `kcpp-android-builder:v0.7`, recreated by `scripts/build-android-builder`
- Builder definition: `containers/android-builder/Dockerfile`
- Pinned base image: `ghcr.io/snapdragon-toolchain/arm64-android:v0.7@sha256:91714433626f0d94a926538a1e46ec43756c5b8e3262b91b95df1e812940aed1`
- Original verified builder image ID: `sha256:e2ebf75d43b604755ef47c577a93488fa898e8d82708c76f0e60c0114182479b`
- Android NDK: r29, API 35
- Qualcomm Hexagon SDK: 6.6.0.0, tools 19.0.07
- GNU Make Debian package: `4.4.1-2`

The complete machine-readable toolchain contract is in `containers/android-builder/requirements.env`. The pinned base image contains the Android NDK and Qualcomm Hexagon SDK; the repository-owned Dockerfile adds only the missing build prerequisite.

For upstream `scripts/snapdragon/run.py` on Windows, prepend `C:/tools/platform-tools` to `PATH`. The script launches `adb` by executable name even when an enclosing shell command used an absolute `adb.exe` path; without that PATH entry, it fails locally with `FileNotFoundError: [WinError 2]` before contacting the phone.

## Build

Run from the repository root on the Windows build host. The first command recreates and validates the pinned builder image; the second performs a clean Android AArch64 build:

```bash
scripts/build-android-builder
scripts/build-android-artifacts
```

`BUILD_JOBS` may override the default parallelism of 8. Explicit make targets may be passed to the artifact script, for example:

```bash
BUILD_JOBS=4 scripts/build-android-artifacts koboldcpp_snapdragon
```

The host script mounts the current Git worktree at `/src`; it does not depend on a fixed `C:/repos/...` checkout path. The in-container script validates the NDK compiler, archiver, and Hexagon SDK before running `make clean`. It prints SHA-256 checksums for every requested standard deliverable that exists.
The compiled deliverables are:

- `mainsnapdragon`: llama-compatible CLI with CPU, OpenCL, and HTP registered together.
- `qwen3ttssnapdragon`: direct Qwen3-TTS CLI with independently selectable transformer, decoder/vocoder, and encoder backends.
- `koboldcpp_snapdragon.so`: KoboldCpp's loadable server library with CPU, OpenCL, and HTP in one backend registry.

A clean build using the command above was verified to remove both custom targets before compilation and then reproduce these SHA-256 hashes:

- `mainsnapdragon`: `a39b18590eac82f6f359450244649a133de80a57928de785314494c17589c144`
- `qwen3ttssnapdragon`: `20c7237f576e5a1b7d98dab74316f80cb123a3eca704666bfbab15409771c2fc`
- `koboldcpp_snapdragon.so`: `966332272f0f4fb716bca21ac2635ea46f5cdf450ef16f983ec6dbbbe11c9983`

The deployed Termux service library, installed as `koboldcpp_default.so`, matches the `koboldcpp_snapdragon.so` hash above exactly. The two standalone CLI hashes are clean-build evidence; those CLIs are not installed in the persistent service directory.

The command-provider-compatible wrapper is `scripts/delilah-qwen3-tts-snapdragon`. Install it as `~/.local/bin/delilah-qwen3-tts-snapdragon`; it accepts `--text-file INPUT --out OUTPUT`. The wrapper was exercised on-device and produced the same byte-identical validated WAV as the direct hybrid test. It is intentionally not selected as Hermes's active provider while xAI remains the accepted voice path.

## Runtime isolation contract

Use the acceleration environment only for the launched process. Do not export it globally.

```bash
ROOT="$HOME/.local/opt/koboldcpp-snapdragon"
HTP="$HOME/.local/opt/llama-htp-sm8850/lib"

LD_LIBRARY_PATH="$ROOT:$PREFIX/opt/vendor/lib:$HTP"
ADSP_LIBRARY_PATH="$HTP"
GGML_HEXAGON_DEVICES="HTP0:0"
GGML_HEXAGON_HOSTBUF=0
GGML_OPENCL_KERNEL_CACHE_DIR="$HOME/.cache/koboldcpp/opencl"
```

Critical constraint: do **not** add `$PREFIX/lib` to `LD_LIBRARY_PATH`. That selects Termux's incompatible `libbinder_ndk` stub and causes `ggml_opencl: platform IDs not available`. The private `libc++_shared.so` in `$ROOT`, the vendor OpenCL bridge in `$PREFIX/opt/vendor/lib`, and the HTP runtime in `$HTP` are sufficient.

## Backend enumeration acceptance check

With the runtime environment above, `mainsnapdragon --list-devices` must expose both:

```text
GPUOpenCL: QUALCOMM Adreno(TM) 840
HTP0:0: Hexagon
```

This was verified on the stock, unrooted Fold 8 Ultra.

## Termux service

The deployed service consists of:

- `~/.local/opt/koboldcpp-snapdragon/koboldcpp.py`
- `~/.local/opt/koboldcpp-snapdragon/koboldcpp_default.so` (the combined Snapdragon library)
- `~/.local/opt/koboldcpp-snapdragon/libc++_shared.so`
- `~/.local/opt/koboldcpp-snapdragon/embd_res/klite.embd` (KoboldAI Lite UI)
- `~/.config/koboldcpp/gemma4-e4b-opencl.kcpps` (model and inference settings)
- `~/.local/bin/koboldcpp-snapdragon-run`
- `~/.local/bin/koboldcpp-snapdragon-service`

The runner contains only the runtime-isolation contract and loads settings from the `.kcpps` file. `KCPP_CONFIG` may select another `.kcpps` profile. The default profile runs Gemma 4 E4B with a 16,384-token context on `GPUOpenCL`, with all 43 model layers offloaded. It loads the proven `whisper-base.en-f16.bin` in the embedded STT slot and `Kokoro_no_espeak_Q4.gguf` in the TTS slot; `embd_res/kokoro_ipa.embd` must be installed beside the runtime. The service manager supports `start`, `stop`, `restart`, `status`, and `log`; its PID and log live under `~/.local/state/koboldcpp-snapdragon/`. It uses `nohup` so the service survives the SSH session, and the runner forwards termination to the Python server and releases its Termux wake lock. This is session persistence, not boot persistence: no Termux:Boot, `runit`, shell-startup, or cron entry currently starts KoboldCpp after an Android reboot.

The installed combined service was verified through the HTTP API on all relevant paths:

- OpenCL: the OuteTTS test model loaded with 17/17 layers on `QUALCOMM Adreno(TM) 840`; `/api/v1/generate` returned HTTP JSON.
- HTP: the same service opened a Hexagon v81 session, enumerated OpenCL/HTP/RPC/CPU, offloaded one layer to `HTP0:0`, and returned HTTP JSON.
- Preferred E4B: the service loaded the abliterated E4B model with `GPUOpenCL`, identified the physical `QUALCOMM Adreno(TM) 840`, and reported `offloaded 43/43 layers to GPU`. `/api/v1/generate` returned HTTP 200 with `FULL_OFFLOAD_OK`.
- Embedded TTS: Kokoro Q4 produced a valid mono 24 kHz WAV through `/v1/audio/speech`.
- Embedded STT: OpenCL Base.en F16 transcribed that WAV through `/v1/audio/transcriptions`; the service remained running after inference.
- KoboldAI Lite: `/` returned HTTP 200 with the complete 1,759,957-byte embedded UI.

The persistent instance listens only on `127.0.0.1:5001`; its `.kcpps` profile sets a 16,384-token context and full E4B offload on Adreno OpenCL. The fixed-output API smoke test proves launchability, full layer placement, and basic deterministic generation; it does not by itself establish broad model-quality equivalence.

## Whisper Base.en Q4_0 OpenCL limitation

The legacy `ggml-base.en-q4_0.bin` artifact is valid, but it must not be selected in the production Snapdragon build. Controlled tests used the same PCM WAV and model file:

| Backend/build | Result | Elapsed |
|---|---|---:|
| CPU | `The hardware speech test day is working.` | 3 s |
| Production OpenCL, SOA + Adreno kernels | empty transcript | 2 s |
| OpenCL, SOA + generic kernels | empty transcript | 4 s |
| Diagnostic OpenCL, AOS + generic kernels | `The hardware speech test day is working.` | 4 s |

This isolates the semantic failure to the OpenCL struct-of-arrays quantized-weight path enabled by `GGML_OPENCL_SOA_Q`, not the model conversion, HTTP endpoint, Hermes adapter, audio normalization, Qualcomm OpenCL driver, or Adreno-specialized kernels. The response remains valid HTTP JSON (`{"text": ""}`), so Hermes treats it as silence and immediately resumes listening.

The diagnostic AOS build required compile guards around SOA-only optimized code and is not a production replacement: disabling SOA globally also changes the quantized Gemma path. Keep Whisper Base.en F16 active until the Q4_0 SOA upload/layout path is corrected and validated without regressing Gemma.

## HTP proof

The imported backend passed upstream `MUL_MAT` conformance on the physical HTP: 570/570 iterations.

The combined KoboldCpp binary then executed a complete 17-layer OuteTTS GGUF graph on `HTP0:0`:

| Offloaded layers | Prompt rate | Decode rate | Total |
|---:|---:|---:|---:|
| 2 | — | — | 380.88 ms |
| 4 | 22.73 tok/s | — | 367.74 ms |
| 8 | 31.30 tok/s | — | 273.14 ms |
| 16 | 127.57 tok/s | — | 79.58 ms |
| 17/17 | 215.92 tok/s | 41.32 tok/s | 57.55 ms |

This proves actual GGUF graph execution on Hexagon HTP, not merely device enumeration.

## Qwen3-TTS backend allocation

The direct Qwen3-TTS executable accepts:

- `QWEN3_TTS_BACKEND`: fallback backend for all components
- `QWEN3_TTS_TRANSFORMER_BACKEND`
- `QWEN3_TTS_DECODER_BACKEND`
- `QWEN3_TTS_ENCODER_BACKEND`

The best verified allocation for the 0.6B Q5_K transformer plus MXFP4 tokenizer/vocoder is:

```bash
QWEN3_TTS_TRANSFORMER_BACKEND=CPU
QWEN3_TTS_DECODER_BACKEND=GPUOpenCL
QWEN3_TTS_ENCODER_BACKEND=CPU
```

Identical smoke input: `Hello, Robert.`, greedy decoding, 32 maximum audio tokens, 2.54 seconds of generated 24 kHz audio.

| Allocation | Model load | Code generation | Vocoder | Synthesis total | Result |
|---|---:|---:|---:|---:|---|
| CPU only | 1.322 s | 3.097 s | 8.431 s | 11.528 s | valid WAV |
| OpenCL only | 3.209 s | 12.141 s | 4.645 s | 16.786 s | valid WAV |
| HTP only | 1.213 s | 24.436 s | 5.044 s | 29.480 s | valid WAV |
| CPU transformer + OpenCL vocoder, run 1 | 1.350 s | 3.028 s | 3.938 s | 6.966 s | valid WAV |
| CPU transformer + OpenCL vocoder, run 2 | 1.406 s | 3.035 s | 3.325 s | **6.360 s** | valid WAV |

The two hybrid runs produced byte-identical 121,814-byte WAV files with SHA-256 `22d21439006c4ce42f2484c99a50e06530ba604d410abe4bc6cbfb6ed7858854`.

Conclusion: the Qwen autoregressive transformer is too serial and dispatch-heavy for the current OpenCL/HTP implementations. The vocoder is the useful Adreno target. Component-level backend selection reduces synthesis from 11.528 seconds to 6.360 seconds, but RTF 2.507 remains slower than conversational real time. A persistent process can remove repeated model loading but cannot by itself remove the 6.36-second synthesis cost.

## Gemma 4 E4B-it result

Tested model:

`Huihui-gemma-4-E4B-it-qat-q4_0-unquantized-abliterated.i1-Q4_0.gguf`

Deterministic 9-token prompt plus 8 generated tokens, context 256, batch 32:

| Backend allocation | Prompt rate | Decode rate | Total inference | Correctness |
|---|---:|---:|---:|---|
| CPU | 5.83 tok/s | 5.79 tok/s | 2.759 s | correct |
| HTP, 8 layers | 0.52 tok/s | 3.84 tok/s | 18.982 s | correct |
| OpenCL, 8 layers | 0.52 tok/s | 4.42 tok/s | 18.795 s | incorrect (`<unused50>` tokens) |

The earlier 8-layer OpenCL result was not representative of the final full-offload configuration. A later live KoboldCpp server run offloaded 43/43 layers to the physical Adreno 840 and returned the requested fixed text through the HTTP API. That supersedes the earlier result for launchability and basic correctness, while the historical partial-offload result remains recorded because broader semantic and sustained-performance validation has not yet been performed.

## Current operating decision

- Keep Hermes STT and TTS on xAI while this runtime remains experimental.
- Do not reactivate or depend on the retired standalone Whisper installation.
- Run the preferred Gemma 4 E4B model with all 43 layers on Adreno OpenCL by default.
- Use HTP for compatible dense GGUF graphs only after a model-specific correctness and latency gate.
- Use OpenCL for the Qwen3-TTS vocoder; do not offload its autoregressive transformer.
- Do not activate local Qwen3-TTS as the default until a persistent implementation reaches conversational latency and passes a listening test.
