{
  description = "Description for the project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    devenv.url = "github:cachix/devenv";
    nix2container.url = "github:nlewo/nix2container";
    nix2container.inputs.nixpkgs.follows = "nixpkgs";
    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
      ];
      systems = [ "x86_64-linux" "i686-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];

      perSystem = { config, self', inputs', pkgs, system, lib, ... }:
        let
          poetry = pkgs.symlinkJoin {
            name = "poetry";
            paths = [ pkgs.poetry ];
            buildInputs = [ pkgs.makeWrapper ];
            postBuild = ''
              wrapProgram $out/bin/poetry --set LD_LIBRARY_PATH ${lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
              ]}
            '';
          };
        in
        {
          devenv.shells.default = {
            name = "llma_index";
            packages = with pkgs; [ gcc ] ++ [ poetry ];
          };
        };
    };
}
