with import <nixpkgs> { };
pkgs.mkShell {
  name = "example-uv-workflow"; # change this to something more catchy
  # Here is where you will add all the libraries required by your native modules
  # You can use the following one-liner to find out which ones you need.
  # Just make sure you have `gcc` installed.
  # `find .venv/ -type f -name "*.so" | xargs ldd | grep "not found" | sort | uniq`

  NIX_LD_LIBRARY_PATH =
    lib.makeLibraryPath [
      stdenv.cc.cc # libstdc++
      zlib # libz (for numpy)
      libuv

      # All the cuda stuff for torch
      cudaPackages.cuda_cudart
      cudaPackages.cudnn
      cudaPackages.cudatoolkit
      cudaPackages.cuda_nvrtc
      cudaPackages.cuda_cupti
    ]
    + ":/run/opengl-driver/lib:/run/opengl-driver-32/lib"; # Adds the global driver path;

  NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";

  packages = with pkgs; [
    uv
  ];

}
