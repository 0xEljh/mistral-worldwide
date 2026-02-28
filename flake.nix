{
  description = "mistral-worldwide dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python312
          uv
          cacert

          git
          gcc
          pkg-config
          cmake
          ninja
          findutils

          # Useful for local GGUF tooling and conversion workflows.
          llama-cpp
        ];

        shellHook = ''
          # WSL CUDA driver (libcuda.so) lives outside the Nix store.
          if [ -d /usr/lib/wsl/lib ]; then
            export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
          fi

          # Triton can fail to resolve libcuda via ldconfig on Nix systems.
          if [ -z "$TRITON_LIBCUDA_PATH" ]; then
            for d in /usr/lib/wsl/lib /usr/lib64-nvidia /run/opengl-driver/lib /run/opengl-driver-32/lib; do
              if [ -e "$d/libcuda.so.1" ]; then
                export TRITON_LIBCUDA_PATH="$d"
                break
              fi
            done
          fi

          if [ -d /usr/lib64-nvidia ]; then
            export LD_LIBRARY_PATH="/usr/lib64-nvidia:$LD_LIBRARY_PATH"
          fi

          if [ -d /usr/local/cuda/lib64 ]; then
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
          fi

          if [ -z "$SSL_CERT_FILE" ]; then
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
          fi

          # Convenience symlinks for llama.cpp binaries in-repo.
          LLAMA_CPP_DIR="$PWD/llama.cpp"
          mkdir -p "$LLAMA_CPP_DIR"
          for bin in llama-quantize llama-cli llama-server llama-gguf-split; do
            if [ -e "${pkgs.llama-cpp}/bin/$bin" ]; then
              ln -sf "${pkgs.llama-cpp}/bin/$bin" "$LLAMA_CPP_DIR/$bin"
            fi
          done

          echo "Nix dev shell ready. Next: uv sync"
        '';
      };
    };
}
