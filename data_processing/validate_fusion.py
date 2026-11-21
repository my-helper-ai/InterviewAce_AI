import pandas as pd
from pathlib import Path


CLEAN_DIR = Path("data/cleaned")
FUSION_FILE = Path(
    "data/fusion/InterviewAce_horizontal_fusion.csv"
)


print("=" * 70)
print("FUSION VALIDATION")
print("=" * 70)


# ---------------------------------------------------------
# LOAD CLEANED DATA
# ---------------------------------------------------------

stack_files = list(CLEAN_DIR.glob("stackexchange_*.csv"))
github_files = list(CLEAN_DIR.glob("github_*.csv"))
remotive_files = list(CLEAN_DIR.glob("remotive_*.csv"))


stack = pd.concat(
    [pd.read_csv(f) for f in stack_files],
    ignore_index=True
)

github = pd.concat(
    [pd.read_csv(f) for f in github_files],
    ignore_index=True
)

remotive = pd.concat(
    [pd.read_csv(f) for f in remotive_files],
    ignore_index=True
)


# ---------------------------------------------------------
# LOAD FUSED DATA
# ---------------------------------------------------------

fused = pd.read_csv(FUSION_FILE)


# ---------------------------------------------------------
# BASIC COUNTS
# ---------------------------------------------------------

print("\nSOURCE RECORD COUNTS")
print("-" * 50)

print(f"Stack Exchange : {len(stack)}")
print(f"GitHub         : {len(github)}")
print(f"Remotive       : {len(remotive)}")

print(f"\nFused rows     : {len(fused)}")
print(f"Fused columns  : {len(fused.columns)}")


# ---------------------------------------------------------
# TECHNOLOGY KEYS
# ---------------------------------------------------------

stack_keys = set(stack["technology"].dropna())
github_keys = set(github["technology"].dropna())
remotive_keys = set(remotive["technology"].dropna())
fused_keys = set(fused["technology"].dropna())


print("\nTECHNOLOGY KEY ANALYSIS")
print("-" * 50)

print(f"Stack Exchange technologies : {len(stack_keys)}")
print(f"GitHub technologies         : {len(github_keys)}")
print(f"Remotive technologies       : {len(remotive_keys)}")
print(f"Fused technologies          : {len(fused_keys)}")


# ---------------------------------------------------------
# MATCHING KEYS
# ---------------------------------------------------------

common_all = (
    stack_keys
    & github_keys
    & remotive_keys
)

stack_github = stack_keys & github_keys
stack_remotive = stack_keys & remotive_keys
github_remotive = github_keys & remotive_keys


print("\nMATCHING TECHNOLOGIES")
print("-" * 50)

print(
    "Stack Exchange + GitHub + Remotive :",
    len(common_all)
)

print(
    "Stack Exchange + GitHub            :",
    len(stack_github)
)

print(
    "Stack Exchange + Remotive          :",
    len(stack_remotive)
)

print(
    "GitHub + Remotive                  :",
    len(github_remotive)
)


# ---------------------------------------------------------
# UNMATCHED KEYS
# ---------------------------------------------------------

print("\nUNMATCHED TECHNOLOGIES")
print("-" * 50)

print("\nStack Exchange only:")
print(sorted(stack_keys - github_keys - remotive_keys))

print("\nGitHub only:")
print(sorted(github_keys - stack_keys - remotive_keys))

print("\nRemotive only:")
print(sorted(remotive_keys - stack_keys - github_keys))


# ---------------------------------------------------------
# DUPLICATE FUSION KEYS
# ---------------------------------------------------------

print("\nDUPLICATE KEYS IN FUSED DATA")
print("-" * 50)

duplicates = fused[
    fused["technology"].duplicated(keep=False)
]

print(f"Duplicate technology rows: {len(duplicates)}")

if len(duplicates) > 0:
    print(duplicates[["technology"]].to_string(index=False))
else:
    print("No duplicate technology keys found.")


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

print("\nMISSING VALUES IN FUSED DATA")
print("-" * 50)

missing = fused.isnull().sum()

print(missing[missing > 0])


# ---------------------------------------------------------
# SOURCE CONTRIBUTION
# ---------------------------------------------------------

print("\nSOURCE CONTRIBUTION")
print("-" * 50)

print(
    "\nStack Exchange rows contributing:",
    (fused["stack_question_count"] > 0).sum()
)

print(
    "GitHub rows contributing:",
    (fused["github_repository_count"] > 0).sum()
)

print(
    "Remotive rows contributing:",
    (fused["remotive_job_count"] > 0).sum()
)


# ---------------------------------------------------------
# FINAL KEY LIST
# ---------------------------------------------------------

print("\nFINAL FUSED TECHNOLOGIES")
print("-" * 50)

print(
    fused["technology"]
    .sort_values()
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)