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
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.rclone # data_uploader.py / data_sync.py — shared cloud sync
          ];

          # Headless training: let matplotlib write predictions.png without
          # needing an X display.
          MPLBACKEND = "Agg";

          shellHook = ''
            echo "Sensa dev environment  —  Python ${python.version} (training-only)"
            echo "  $(rclone version | head -n1)"
            echo
            echo "Workflow A — public EPA AQS data (run from tools/pytorch_calibration/):"
            echo "  python fetch_public_data.py        # download + pair AQS data"
            echo "  python train_public.py             # train 3-feature model"
            echo
            echo "Workflow B — local SEN55 + BAM co-location:"
            echo "  python prepare_data.py             # pair SEN55 .pkl with BAM CSV"
            echo "  python main.py --no-export         # train 8-feature model"
            echo
            echo "Data collection / sharing (run from tools/):"
            echo "  python uart_logger.py --port /dev/ttyUSB0"
            echo "  python data_uploader.py / data_sync.py"
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
