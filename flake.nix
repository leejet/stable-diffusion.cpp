{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        name = "stable-diffusion.cpp";
        src = ./.;
        meta.mainProgram = "sd";
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
        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config ];
        cudatoolkit_joined = with pkgs; symlinkJoin {
          # HACK(Green-Sky): nix currently has issues with cmake findcudatoolkit
          # see https://github.com/NixOS/nixpkgs/issues/224291
          # copied from jaxlib
          name = "${cudaPackages.cudatoolkit.name}-merged";
          paths = [
            cudaPackages.cudatoolkit.lib
            cudaPackages.cudatoolkit.out
          ] ++ lib.optionals (lib.versionOlder cudaPackages.cudatoolkit.version "11") [
            # for some reason some of the required libs are in the targets/x86_64-linux
            # directory; not sure why but this works around it
            "${cudaPackages.cudatoolkit}/targets/${system}"
          ];
        };
        cmakeFlags = [ ];
        #cmakeFlags = [ "-DBUILD_SHARED_LIBS=ON" "-DCMAKE_SKIP_BUILD_RPATH=ON" ];
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          inherit name src meta nativeBuildInputs;
          buildInputs = osSpecific;
          #cmakeFlags = cmakeFlags
            #++ (if isAarch64 && isDarwin then [
            #"-DCMAKE_C_FLAGS=-D__ARM_FEATURE_DOTPROD=1"
            #"-DLLAMA_METAL=ON"
          #] else [
            #"-DLLAMA_BLAS=ON"
            #"-DLLAMA_BLAS_VENDOR=OpenBLAS"
          #]);
        };
        packages.cuda = pkgs.stdenv.mkDerivation {
          inherit name src meta nativeBuildInputs;
          buildInputs = with pkgs; buildInputs ++ [ cudatoolkit_joined ];
          cmakeFlags = cmakeFlags ++ [
            "-DSD_CUBLAS=ON" # for the pr
            "-DGGML_CUBLAS=ON" # for current master
          ];
        };
        apps.sd = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/sd";
        };
        apps.default = self.apps.${system}.sd;
      });
}
