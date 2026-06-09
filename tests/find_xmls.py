import glob
files = glob.glob("d:/road_hazard_datasets/**/*.xml", recursive=True)
print("Found XML files:", len(files))
if files:
    print("Example:", files[0])
