{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        name = "stable-diffusion.cpp";
        src = ./.;
        meta.mainProgram = "sd";
        stdenv = (pkgs.stdenvAdapters.keepDebugInfo pkgs.stdenv);
        inherit (pkgs.stdenv) isAarch32 isAarch64 isDarwin;
        buildInputs = with pkgs; [ ];
        osSpecific = with pkgs; buildInputs ++ (
          if isAarch64 && isDarwin then
            with pkgs.darwin.apple_sdk_11_0.frameworks; [
              Accelerate
              MetalKit
            ]
          else if isAarch32 && isDarwin then
            with pkgs.darwin.apple_sdk.frameworks; [
              Accelerate
              CoreGraphics
              CoreVideo
            ]
          else if isDarwin then
            with pkgs.darwin.apple_sdk.frameworks; [
              Accelerate
              CoreGraphics
              CoreVideo
            ]
          else
            with pkgs; [ openblas ]
        );
        pkgs = import nixpkgs { inherit system; };
        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config git ];
        #cudatoolkit_joined = with pkgs; symlinkJoin {
        #  # HACK(Green-Sky): nix currently has issues with cmake findcudatoolkit
        #  # see https://github.com/NixOS/nixpkgs/issues/224291
        #  # copied from jaxlib
        #  name = "${cudaPackages.cudatoolkit.name}-merged";
        #  paths = [
        #    cudaPackages.cudatoolkit.lib
        #    cudaPackages.cudatoolkit.out
        #  ] ++ lib.optionals (lib.versionOlder cudaPackages.cudatoolkit.version "11") [
        #    # for some reason some of the required libs are in the targets/x86_64-linux
        #    # directory; not sure why but this works around it
        #    "${cudaPackages.cudatoolkit}/targets/${system}"
        #  ];
        #};
        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
          #"-DCMAKE_C_FLAGS:STRING=-Og"
          #"-DCMAKE_CXX_FLAGS:STRING=-Og"

          "-DGGML_NATIVE=OFF"
          "-DGGML_AVX=ON"
          "-DGGML_AVX2=ON"
          "-DGGML_FMA=ON"
          "-DGGML_F16C=ON"

          #"-DBUILD_SHARED_LIBS=ON"
          "-DCMAKE_SKIP_BUILD_RPATH=ON"
        ];
      in
      {
        packages.default = stdenv.mkDerivation {
          inherit name src meta nativeBuildInputs;
          buildInputs = osSpecific;
        };
        packages.cuda = stdenv.mkDerivation {
          inherit name src meta;
          buildInputs = with pkgs; buildInputs ++ [
            #cudaPackages.cudatoolkit
            cudaPackages.cuda_cccl # <nv/target>

            # A temporary hack for reducing the closure size, remove once cudaPackages
            # have stopped using lndir: https://github.com/NixOS/nixpkgs/issues/271792
            cudaPackages.cuda_cudart
            cudaPackages.libcublas
          ];
          nativeBuildInputs = with pkgs; nativeBuildInputs ++ [
            cudaPackages.cuda_nvcc
            autoAddDriverRunpath
          ];
          cmakeFlags = cmakeFlags ++ [
            "-DSD_CUDA=ON"
            "-DCMAKE_CUDA_ARCHITECTURES=75"
          ];
        };
        apps.sd = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/sd";
        };
        apps.default = self.apps.${system}.sd;
      });
}
