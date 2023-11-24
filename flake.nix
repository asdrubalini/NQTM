{
  description = "NQTM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-20.03";
    nixpkgs-nvidia.url = "https://github.com/nixos/nixpkgs/archive/0fdc7224a24203d9489bc52892e3d6121cacb110.tar.gz";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, nixpkgs-nvidia, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        pkgs-nvidia = import nixpkgs-nvidia {
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            (pkgs.python36.withPackages (ps: with ps; [
              tensorflowWithCuda
              numpy
              scikitlearn
            ]))
          ];

          shellHook = ''
            export CUDA_PATH=${pkgs-nvidia.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs-nvidia.linuxPackages.nvidia_x11}/lib:${pkgs-nvidia.ncurses5}/lib
            export EXTRA_LDFLAGS="-L/lib -L${pkgs-nvidia.linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"
          '';
        };
      }
    );
}
