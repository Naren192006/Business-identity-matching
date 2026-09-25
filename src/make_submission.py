"""Build the final submission zip per the challenge structure:

<team>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/ (src/, README.md, requirements.txt)
└── Documentation_template.md
"""
import os
import zipfile

TEAM = "team_name"


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_zip = os.path.join(base, f"{TEAM}_submission.zip")
    files = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("Documentation_template.md", "Documentation_template.md"),
        ("README.md", "code/business_entity_resolution/README.md"),
        ("requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ]
    src_dir = os.path.join(base, "src")
    for fn in os.listdir(src_dir):
        if fn.endswith(".py"):
            files.append((f"src/{fn}", f"code/business_entity_resolution/src/{fn}"))
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, arc in files:
            p = os.path.join(base, rel)
            if not os.path.exists(p):
                print(f"MISSING (skipped): {rel}")
                continue
            z.write(p, arc)
            print(f"+ {arc}")
    print(f"\nwrote {out_zip} "
          f"({os.path.getsize(out_zip)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
