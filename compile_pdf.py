from pathlib import Path
import shutil
from reward_hacking.reporting import compile_pdf_report

exp_dir = Path("runs/v2_exp_20261103_014644_ac0d")
md_path = exp_dir / "REPORT.md"
pdf_path = exp_dir / "REPORT.pdf"

print(f"Compiling PDF from {md_path} to {pdf_path}...")
compile_pdf_report(md_path, pdf_path)
print(f"Successfully created: {pdf_path} ({pdf_path.stat().st_size} bytes)")

# Also copy to root demo_cap and brain artifacts dir
root_pdf = Path("c:/demo_cap/REPORT.pdf")
shutil.copy2(pdf_path, root_pdf)
print(f"Copied to root: {root_pdf}")

brain_pdf = Path("C:/Users/aryad/.gemini/antigravity/brain/ee1d0a6d-77c8-4e61-ad0f-94bc7432074f/REPORT.pdf")
shutil.copy2(pdf_path, brain_pdf)
print(f"Copied to brain: {brain_pdf}")
