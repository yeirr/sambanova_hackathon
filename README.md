# Introduction

A SambaNova GenAI project on human nutrition.

## Features

* Local web ui
* Local llm instance
* Semantic cache
* Natural language QA with knowledge graph(3100 nodes, 2792 relations)
* Multi-agent reasoning with majority vote
* Human nutrition topics(general info, food safety, healthy diet, WHO factsheets)
* Multi-modality(Vision-Language)
  * support:
    + Llama-3.2-11B-Vision-Instruct, 4k ctx
    + Llama-3.2-90B-Vision-Instruct, 4k ctx
    + Llama-3.1-70B-Instruct, 64k ctx
    + Llama-3.1-405B-Instruct, 8k ctx
    + natural resolution and various aspect ratios
    + single and multi-image inputs
    + natural images(tested up to 8k resolution)
    + documents(figures: graph, diagram, charts, tables etc)
    + multi-modal in-context learning(multiple image-instruction-response tuples)
  * use cases:
    + reasoning over complex figures
    + chart understanding, analysis and actionable plan to drive desirable outcomes
    + multi-image instruction following
    + image-to-code
    + natural scene understanding
    + image-text retrieval
    + image captioning
    + mm machine translation
    + image generation
    + visual qa
    + visual commonsense reasoning
    + visual storytelling
    + accessibility
    + content moderation
    + medical imaging analysis
    + robotics and autonomous systems

## Instructions

Run following commands from project root directory.

### Prerequisites

* Requires at least:
  + 16 CPU cores
  + 32GB RAM
  + 1x Nvidia GPU with 24GB VRAM
* Recommended:
  + 24 CPU cores
  + 64GB RAM
  + 1x Nvidia GPU with 40GB VRAM
* Supports Ubuntu LTS(2204/2404) host only.
* Ensure [Docker](https://docs.docker.com/engine/install/) is installed on host machine.
* Ensure [Compose Plugin](https://docs.docker.com/compose/install/#scenario-two-install-the-compose-plugin) is installed on host machine.

* Install python dependencies.

```python
pip install -r requirements.txt
```

### Backend

Initialize local instances.

```bash
docker compose \
  --env-file=.env \
  --profile redis-stack \
  --profile pg16age up -d
```
> Meanwhile wait for 'contest-pg16age' to initialize before importing graph data from CSV files by running mounted SQL script.

```bash
docker exec -it contest-pg16age \
  psql -U postgres \
  -d postgres \
  -f /tmp/load_kg.sql \
  -f /tmp/who_healthy_diet.sql \
  -f /tmp/un_who_healthy_diets.sql \
  -f /tmp/who_food_safety.sql \
  -f /tmp/who_factsheets_part1of4.sql \
  -f /tmp/who_factsheets_part2of4.sql \
  -f /tmp/who_factsheets_part3of4.sql \
  -f /tmp/who_factsheets_part4of4.sql \
  -f /tmp/postprocess_kg.sql
```

```bash
bash scripts/run_vllm
```
> Wait for backend instances to be initialized before running frontend.

### Frontend

Initialize UI.

```python
SNOVA_API_KEY="<REPLACE_WITH_API_KEY>" streamlit run main.py
```

## Limitations

* Requires warmup as initial inference requires building of finite state machine(FSM) for structured output.
* Lacks data persistence on host filesystem.
* Knowledge graph data modeling is unoptimized.
* Cypher queries are unoptimized.

## Resources

Information sourced from:

* Academy of Nutrition and Dietetics
* American Cancer Society - Nutrition and Physical Activity
* American Diabetes Association (ADA) - Nutrition and Recipes
* American Gastroenterological Association (AGA) - Nutrition and Healthy Living
* American Heart Association (AHA) - Nutrition Center
* American Journal of Clinical Nutrition
* Centers for Disease Control and Prevention (CDC) - Nutrition
* European Food Safety Authority (EFSA) - Nutrition
* Harvard School of Public Health - Nutrition Source
* Journal of the Academy of Nutrition and Dietetics
* National Academy of Sciences - Institute of Medicine (IOM) - Food and Nutrition Board
* National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) - Nutrition and Diet
* National Institute on Aging (NIA) - Nutrition and Aging
* National Institutes of Health (NIH) - Office of Dietary Supplements
* The Lancet - Nutrition, Health and Medicine
* United States Department of Agriculture (USDA) - ChooseMyPlate
* World Health Organization (WHO) - Nutrition

## Demo

Video at [Instagram](https://www.instagram.com/reel/DCoFsV_ONhk/?utm_source=ig_web_copy_link).
