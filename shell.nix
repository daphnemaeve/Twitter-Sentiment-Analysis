with import <nixpkgs> {};
mkShell {
  name = "python";
  venvDir = ".venv";

  buildInputs = with python39Packages; [
    venvShellHook
  ];

  postShellHook = ''
    pip install -r requirements.txt
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
