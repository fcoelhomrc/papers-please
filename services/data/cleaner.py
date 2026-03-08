# NOTE:
# If pdf file is deleted from disk,
# but referenced in the Objects table
# We should invalidate it by
# 1. Create a manual, temporary index of existing filenames
# 2. Query objects (paginate it!)
# 3. Compare returned data with existing filenames
# 4. Collect ids to drop from object tables
# 5. Send alert / trigger a new fetch job
