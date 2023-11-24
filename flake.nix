{
  description = "NQTM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-20.03";
  };

  # Flake outputs
  outputs = { self, nixpkgs, }:
    let
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux
        "x86_64-darwin" # 64-bit Intel macOS
        "aarch64-darwin" # 64-bit ARM macOS
      ];

      forAllSystems = f:
        nixpkgs.lib.genAttrs allSystems
        (system: f { pkgs = import nixpkgs { inherit system; }; });
    in {
      # Development environment output
      devShells = forAllSystems ({ pkgs }: {
        default = pkgs.mkShell {
          buildInputs = [
            (pkgs.python36.withPackages (ps:
              with ps; [
                numpy
                tensorflow
                scikitlearn
            ]))
          ];
        };
      });
    };
}
