{ pkgs ? import <nixpkgs> {} }:

let
  libs = with pkgs; [ stdenv.cc.cc.lib zlib glib libglvnd ];
in
pkgs.mkShell {
  packages = with pkgs; [ 
    python311 
    python311Packages.pip 
    ollama 
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libs}:$LD_LIBRARY_PATH"
    [ ! -d .venv ] && python -m venv .venv
    source .venv/bin/activate
    pip install psutil matplotlib gputil
  '';
}
