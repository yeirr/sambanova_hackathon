CREATE VIEW global_kg AS
SELECT * FROM ag_catalog.cypher('general', $$
    MATCH p=(a)-[b]->(c)
    WITH a.type as node_1, label(b) as relation, c.type as node_2
    RETURN DISTINCT node_1, relation, node_2
    LIMIT 1000
$$) AS (node_1 agtype, relation agtype, node_2 agtype)
UNION ALL
SELECT * FROM ag_catalog.cypher('healthy_diet_part1of2', $$
    MATCH p=(a)-[b]->(c)
    WITH a.type as node_1, label(b) as relation, c.type as node_2
    RETURN DISTINCT node_1, relation, node_2
    LIMIT 1000
$$) AS (node_1 agtype, relation agtype, node_2 agtype)
UNION ALL
SELECT * FROM ag_catalog.cypher('healthy_diet_part2of2', $$
    MATCH p=(a)-[b]->(c)
    WITH a.type as node_1, label(b) as relation, c.type as node_2
    RETURN DISTINCT node_1, relation, node_2
    LIMIT 1000
$$) AS (node_1 agtype, relation agtype, node_2 agtype)
UNION ALL
SELECT * FROM ag_catalog.cypher('food_safety', $$
    MATCH p=(a)-[b]->(c)
    WITH a.type as node_1, label(b) as relation, c.type as node_2
    RETURN DISTINCT node_1, relation, node_2
    LIMIT 1000
$$) AS (count_node_1 agtype, count_relation agtype, node_2 agtype)
UNION ALL
SELECT * FROM ag_catalog.cypher('who_factsheets_part1of4', $$
    MATCH p=(a)-[b]->(c)
    WITH a.type as node_1, label(b) as relation, c.type as node_2
    RETURN DISTINCT node_1, relation, node_2
    LIMIT 1000
$$) AS (node_1 agtype, relation agtype, node_2 agtype)
UNION ALL
SELECT * FROM ag_catalog.cypher('who_factsheets_part2of4', $$
    MATCH p=(a)-[b]->(c)
    WITH a.type as node_1, label(b) as relation, c.type as node_2
    RETURN DISTINCT node_1, relation, node_2
    LIMIT 1000
$$) AS (node_1 agtype, relation agtype, node_2 agtype)
UNION ALL
SELECT * FROM ag_catalog.cypher('who_factsheets_part3of4', $$
    MATCH p=(a)-[b]->(c)
    WITH a.type as node_1, label(b) as relation, c.type as node_2
    RETURN DISTINCT node_1, relation, node_2
    LIMIT 1000
$$) AS (node_1 agtype, relation agtype, node_2 agtype)
UNION ALL
SELECT * FROM ag_catalog.cypher('who_factsheets_part4of4', $$
    MATCH p=(a)-[b]->(c)
    WITH a.type as node_1, label(b) as relation, c.type as node_2
    RETURN DISTINCT node_1, relation, node_2
    LIMIT 1000
$$) AS (node_1 agtype, relation agtype, node_2 agtype);

SELECT node_1, relation, node_2 FROM global_kg ORDER BY node_1;
