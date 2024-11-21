from typing import Any, Dict, List

prebuilt_data_list: List[Dict[str, Any]] = [
    # General.
    {
        "query": "tell me about autoimmune disorders",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Autoimmune Disorders" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "general"},
    },
    {
        "query": "tell me about dysbiosis",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Dysbiosis" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "general"},
    },
    {
        "query": "tell me about excessive sugar consumption",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Excessive Sugar Consumption" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "general"},
    },
    {
        "query": "tell me about gut microorganisms",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Gut Microorganisms" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "general"},
    },
    {
        "query": "tell me about high dietary fiber consumption",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "High Dietary Fiber Consumption" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "general"},
    },
    # Food Safety.
    {
        "query": "tell me about antimicrobial agents",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Antimicrobial Agents" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "food safety"},
    },
    {
        "query": "tell me about foodborne illnesses",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Foodborne Illnesses" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "food safety"},
    },
    {
        "query": "tell me about climate change",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Climate Change" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "food safety"},
    },
    {
        "query": "tell me about the sources of heavy metals in my diet",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Heavy Metals" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "food safety"},
    },
    {
        "query": "tell me about listeria, mycotoxins, norovirus, prions and salmonella",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Listeria" OR node_1 = "Mycotoxins" OR node_1 = "Norovirus" OR node_1 = "Prions" OR node_1 = "Salmonella" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "food safety"},
    },
    # Healthy Diet.
    {
        "query": "tell me about added sugars in my diet",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Added Sugars" OR node_1 = "Tooth Decay" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "explain what a balanced diet is to a five year old",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Balanced Diet" OR node_1 = "Unhealthy Dietary Patterns" OR node_1 = "Unhealthy Eating" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "how does breast feeding contribute to a infant's healthy diet",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Breast feeding" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "explain role of dietary fats as part of healthy diet",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Dietary Fats" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "explain role fresh produce as part of healthy diet",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Fresh Produce" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "explain role of sodium chloride(salt) found in human diet",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Sodium Chloride" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "explain role of ultra processed food",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Ultra-Processed Foods" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "explain role of whole foods part of balanced diet",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Whole Foods" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "explain role of whole grain cereal as part of balanced diet",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Whole Grain Cereals" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "what are the key characteristics of healthy diets",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Key Characteristics of Healthy Diets" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "what are the benefits of keeping a healthy diet",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Healthy Diets" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    {
        "query": "what are healthy eating patterns",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Healthy Eating Patterns" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "healthy diet"},
    },
    # WHO Factsheets.
    {
        "query": "how is anaemia related to poor diet?",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Anaemia" OR node_1 = "Global Anaemia Alliance" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
    {
        "query": "what are the adverse effects of alcohol intake",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Alcohol Intake" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
    {
        "query": "how does artificial sweeteners affect me",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Artificial Sweeteners" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
    {
        "query": "does childhood obesity affect adult obesity",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Childhood Obesity" OR node_1 = "Obesity" OR node_1 = "Obesity Prevention" OR node_1 = "Overweight and Obesity" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
    {
        "query": "what are the risk factors for colorectal cancer",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Colorectal Cancer" OR node_1 = "Colorectal Cancer Screening" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
    {
        "query": "what are some good dietary habits",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Dietary Habits" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
    {
        "query": "how does malnutrition affect me",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Malnutrition" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
    {
        "query": "how does physical activity or inactivity affect me",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Physical Activity" OR node_1 = "Physical Inactivity" OR node_1 = "Sedentary Behavior" OR node_1 = "Sedentary Lifestyle" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
    {
        "query": "tell me about diarrheal diseases",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Diarrheal Diseases" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
    {
        "query": "tell me about food additives",
        "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Food Additives" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
        "metadata": {"type": "cypher", "domain": "who factsheets"},
    },
]
