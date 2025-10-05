import weaviate
c = weaviate.connect_to_local()
col = c.collections.get("MedicalDoc")
try:
    agg = col.aggregate.over_all(total_count=True)
    print("Total objects:", agg.total_count)
finally:
    c.close()