{
  description = "NQTM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-20.03";
    nixpkgs-nvidia.url = "https://github.com/nixos/nixpkgs/archive/ace5093e36ab1e95cb9463863491bee90d5a4183.tar.gz";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, nixpkgs-nvidia, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;

          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        pkgs-nvidia = import nixpkgs-nvidia {
          inherit system;

          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        buildInputs = [
          (pkgs.python36.withPackages (ps: with ps; [
            tensorflowWithCuda
            numpy
            scikitlearn
          ]))
        ];

        shellHook = ''
          echo "Fasten your seatbelt"
          export CUDA_PATH=${pkgs-nvidia.cudatoolkit}
          export LD_LIBRARY_PATH=${pkgs-nvidia.linuxPackages.nvidia_x11}/lib:${pkgs-nvidia.ncurses5}/lib
          export EXTRA_LDFLAGS="-L/lib -L${pkgs-nvidia.linuxPackages.nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-I/usr/include"
        '';

      in
      {
        devShells.default = pkgs.mkShell {
	        name = "NQTM";
          inherit buildInputs;
          inherit shellHook;
        };
      }
    );
}
