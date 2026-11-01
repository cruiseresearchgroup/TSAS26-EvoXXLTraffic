"""Write <run_root>/CODE_RECEIPT.json, the source seal that reviewer_fixed_vs_evolving checks before a run: the sha256 of
every tracked file of this checkout and its commit (standard library only).
usage: python make_code_receipt.py <run_root>"""
import hashlib, json, pathlib, subprocess, sys

root = pathlib.Path(sys.argv[1])
source = pathlib.Path(__file__).resolve().parents[3]

def git(*args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True)

files = git("ls-files", "-z").split("\0")[:-1]
seal = {f: hashlib.sha256((source / f).read_bytes()).hexdigest() for f in files}
receipt = dict(code_commit=git("rev-parse", "HEAD").strip(), files=seal)
(root / "CODE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("sealed %d files at %s" % (len(seal), receipt["code_commit"]))
