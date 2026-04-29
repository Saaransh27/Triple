from neo4j import GraphDatabase

URI  = "neo4j+s://f4998694.databases.neo4j.io"
USER = "f4998694"
PASS = "_6jblwHLzd6UViOCdWA7NOhtJwGDLGgyXPiDt8CsspI"

driver = GraphDatabase.driver(URI, auth=(USER, PASS))
with driver.session() as session:
    result = session.run("RETURN 1 AS x")
    print(result.single()["x"])
driver.close()