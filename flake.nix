{
  description = "NQTM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-20.03";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            (pkgs.python36.withPackages (ps: with ps; [
              numpy
              tensorflow
              scikitlearn
            ]))
          ];
        };
      }
    );
}
