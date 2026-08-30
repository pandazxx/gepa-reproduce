{
  description = "gepa-reproduce dev environment: Python, uv, just, Ollama";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSystem = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forEachSystem (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python311
              uv
              just
              ollama
              curl
              git
            ];

            shellHook = ''
              echo "gepa-reproduce devshell ready: $(python3 --version), $(just --version), ollama $(ollama --version 2>/dev/null | head -n1)"
              echo "Next: just setup && just pull && just serve (separate terminal) && just preflight"
            '';
          };
        });
    };
}
