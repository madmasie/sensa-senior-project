{
  description = "Sensa — reproducible dev environment for the PM2.5 calibration ML pipeline (tools/)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Python 3.13 — satisfies tools/pyproject.toml's requires-python
        # ">=3.10,<3.14". nixpkgs ships torch/tensorflow/onnx built against it.
        python = pkgs.python313;

        # Every Python dependency the tools/ scripts need that is available
        # from nixpkgs. torch is the default CPU build: this machine has an
        # AMD GPU (no CUDA), and tools/README.md notes CPU is fast enough
        # (<1 min/epoch) for this small model.
        pythonEnv = python.withPackages (
          ps: with ps; [
            # ── core ── uart_logger.py, data_uploader.py, data_sync.py
            pyserial
            pandas
            numpy
            pyyaml

            # ── ble ── sensa_client.py
            bleak

            # ── ml ── pytorch_calibration training (main.py / train_public.py)
            torch
            onnx
            scikit-learn
            matplotlib

            # ── public data ── fetch_public_data.py (EPA AQS bulk downloads)
            requests
            tqdm
          ]
        );

        # ── Runnable apps ───────────────────────────────────────────────────
        # Each tools/ script is wrapped as a `nix run .#<name>` app. The wrapper
        # locates the repo with `git rev-parse`, so the app works from any
        # directory inside the working tree, and `cd`s into the directory the
        # script expects (its relative data/model paths resolve from there).
        # Output files are written into the live working tree, never the
        # read-only /nix/store copy.
        mkScriptApp =
          { dir, script }:
          let
            wrapper = pkgs.writeShellApplication {
              name = "sensa-${pkgs.lib.removeSuffix ".py" script}";
              runtimeInputs = [ pythonEnv pkgs.rclone pkgs.git ];
              text = ''
                repo_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
                if [ -z "$repo_root" ]; then
                  echo "error: run this from inside the sensa-senior-project git repo" >&2
                  exit 1
                fi
                export MPLBACKEND=Agg
                cd "$repo_root/${dir}"
                exec python ${script} "$@"
              '';
            };
          in
          {
            type = "app";
            program = "${wrapper}/bin/${wrapper.name}";
            meta.description = "Run ${dir}/${script}";
          };
      in
      {
        # `nix run .#<name>` — see tools/README.md "Running with Nix".
        apps = {
          # ── Data collection / sharing (tools/) ──
          uart-logger   = mkScriptApp { dir = "tools"; script = "uart_logger.py"; };
          sensa-client  = mkScriptApp { dir = "tools"; script = "sensa_client.py"; };
          data-uploader = mkScriptApp { dir = "tools"; script = "data_uploader.py"; };
          data-sync     = mkScriptApp { dir = "tools"; script = "data_sync.py"; };

          # ── ML pipeline (tools/pytorch_calibration/) ──
          fetch-public-data = mkScriptApp { dir = "tools/pytorch_calibration"; script = "fetch_public_data.py"; };
          prepare-data      = mkScriptApp { dir = "tools/pytorch_calibration"; script = "prepare_data.py"; };
          train-public      = mkScriptApp { dir = "tools/pytorch_calibration"; script = "train_public.py"; };
          train-local       = mkScriptApp { dir = "tools/pytorch_calibration"; script = "main.py"; };
          paper-figures     = mkScriptApp { dir = "tools/pytorch_calibration"; script = "paper_figures.py"; };
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.rclone # data_uploader.py / data_sync.py — shared cloud sync
          ];

          # Headless training: let matplotlib write figures without an X display.
          MPLBACKEND = "Agg";

          shellHook = ''
            echo "Sensa dev environment  —  Python ${python.version} (training-only)"
            echo "  $(rclone version | head -n1)"
            echo
            echo "Run scripts directly with 'nix run' (works from anywhere in the repo):"
            echo "  nix run .#fetch-public-data        # download + pair AQS data"
            echo "  nix run .#train-public             # train 3-feature model"
            echo "  nix run .#prepare-data             # pair SEN55 .pkl with BAM CSV"
            echo "  nix run .#train-local -- --no-export   # train 8-feature model"
            echo "  nix run .#paper-figures -- --demo  # generate paper figures"
            echo "  nix run .#uart-logger -- --port /dev/ttyUSB0"
            echo "  nix run .#data-uploader / .#data-sync"
            echo "  (nix flake show  lists every app)"
            echo
            echo "Or run the scripts the classic way from inside this shell, e.g."
            echo "  cd tools/pytorch_calibration && python train_public.py"
            echo
            echo "NOTE: TFLite export (--export) needs onnx2tf, which is not in"
            echo "      nixpkgs and is excluded from this shell. To export, run the"
            echo "      step in a separate venv:  python -m venv .export-venv &&"
            echo "      .export-venv/bin/pip install onnx2tf tensorflow-cpu"
          '';
        };
      }
    );
}
