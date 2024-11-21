import asyncio
import base64
import concurrent.futures
import json
import os
import time
import typing
import warnings
from datetime import datetime, timezone

import openai
import psycopg
import streamlit as st
import uvloop
import yaml
from dotenv import dotenv_values
from dspy.predict import aggregation
from dspy.primitives.prediction import Completions
from duckduckgo_search import AsyncDDGS
from duckduckgo_search.exceptions import RatelimitException
from jinja2 import Environment, FileSystemLoader
from langchain_community.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_community.graphs.age_graph import AGEGraph
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatResponse, ChatResponseAsyncGen
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import CompletionUsage
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_extras.bottom_container import bottom

from prebuilt_data import prebuilt_data_list

# General configuration.
warnings.simplefilter("ignore")
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
config = dotenv_values(dotenv_path=".env")

# Postgres DB.
kg = AGEGraph(
    graph_name=config["GRAPH_NAME"],
    conf={
        "database": "postgres",
        "user": "postgres",
        "host": "localhost",
        "port": 5432,
    },
)
# Validate cypher query for relationship directions.
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in kg.structured_schema.get("relationships")
]
cypher_validation = CypherQueryCorrector(corrector_schema)


# Utility functions.
def read_system_templates(
    filepath: str = "./prompts/templates/expert-identity",
    system_type: str = "generic",
) -> typing.List[str]:
    """Read and render system prompt templates for expert identities from directory.

    Example
    =======
        system_type = "generic"
        list_system_prompt = read_system_templates_expert_identity(system_type=system_type)
        print(list_system_prompt[0])

    Returns
    =======
        A list of rendered system prompt templates.
    """
    prompts = []
    environment = Environment(loader=FileSystemLoader(filepath))

    templates = [os.path.join(filepath, file.name) for file in os.scandir(filepath)]
    for idx in templates:
        if system_type in os.path.split(idx)[-1]:
            template = environment.get_template(os.path.split(idx)[1])
            render = template.render()
            prompts.append(yaml.safe_load(render)["template"])

    return prompts


async def call_agent_endpoint(
    query: str,
    agent: str,
    expert_identity: str,
    cypher_statements_parsed: str,
    timeout: int = 60,
) -> str:
    # Parse expert prompt and query for agent inference.
    message = f"""
        [Cypher Statement]
        {cypher_statements_parsed}

        [Identity Background]
        {expert_identity}

        Now given the above identity background, please answer the following query using
        only the facts present in Cypher statements in paragraph form with no conclusion
        or summary: {query}
        """

    agent_messages = []

    try:
        ddgs_agent_message = await AsyncDDGS(timeout=timeout).achat(
            message,
            model=agent,
        )
        agent_messages.append(ddgs_agent_message)
    except RatelimitException as e:
        raise e

    return agent_messages[-1]


async def convert_to_chat_response_generator(data: str) -> ChatResponseAsyncGen:
    """
    Example
    =======
        data = "what is the meaning of life?"
        gen = convert_to_chat_response_generator(data=data)
        async for chunk in gen:
            print(chunk)
    """
    response = ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=int(datetime.now(timezone.utc).timestamp()),
        model=config["MODEL_ID_VLM_90B"],
        usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant",
                    content=data,
                ),
                finish_reason="length",
                index=0,
            )
        ],
    )
    time.sleep(0.025)
    yield ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=data,
        ),
        delta=data,
        raw=response,
    )


async def run_cypher_query(cypher_query: str) -> str:
    cypher_fields = "node_1 agtype, relation agtype, node_2 agtype"

    # Query prebuilt knowledge subgraphs.
    subgraphs_list: typing.List[str] = [
        "general",
        "healthy_diet_part1of2",
        "healthy_diet_part2of2",
        "food_safety",
        "who_factsheets_part1of4",
        "who_factsheets_part2of4",
        "who_factsheets_part3of4",
        "who_factsheets_part4of4",
    ]
    kg_responses: typing.List[tuple[str, str, str]] = []
    for subgraph in subgraphs_list:
        with psycopg.connect(config["DB_URL"]) as conn:
            with conn.cursor() as cur:
                try:
                    # Cypher commands.
                    template = """
                        SELECT * FROM ag_catalog.cypher('{graph_name}', $$ 
                            {cypher_query}
                        $$) AS ({cypher_fields})"""

                    # Mandatory.
                    cur.execute("""SET search_path = ag_catalog, "$user", public""")
                    cur.execute(
                        cypher_validation(
                            template.format(
                                graph_name=subgraph,
                                cypher_query=cypher_query,
                                cypher_fields=cypher_fields,
                            )
                        )
                    )
                    for row in cur:
                        kg_responses.append(row)
                except Exception as e:
                    conn.rollback()

    # Parse kg tuple responses into list of strings.
    # Add double whitespaces to render newline in UI.
    kg_response_parsed = "  \n".join(
        [" ".join(map(str, idx)).replace('"', "") for idx in kg_responses]
    )

    return kg_response_parsed


# Initialization of services.
@st.cache_resource
def init_semantic_cache() -> SemanticCache:
    # Local semantic cache.
    hf = HFTextVectorizer(model="mixedbread-ai/mxbai-embed-large-v1")
    semantic_cache = SemanticCache(
        name="llmcache",
        vectorizer=hf,
        redis_url=config["REDIS_LOCAL_URL"],
        ttl=15,
        distance_threshold=0.1,
    )

    # Populate cache with initial data of query-cypher pairs.
    initial_data_list = [
        {
            "query": "tell me about human gut microbiome",
            "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Gut Microbiome" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
            "metadata": {"type": "cypher", "domain": "general"},
        },
        {
            "query": "explain food safety",
            "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Food Safety" OR node_1 = "Food Security" OR node_1 = "Food Systems" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
            "metadata": {"type": "cypher", "domain": "food safety"},
        },
        {
            "query": "tell me about disease prevention from having a healthy diet",
            "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Fresh Fruits" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
            "metadata": {"type": "cypher", "domain": "healthy diet"},
        },
        {
            "query": "tell me about pesticide residues present in our environment and food",
            "response": 'MATCH p=(a)-[b]->(c) WITH a.type as node_1, label(b) as relation, c.type as node_2 WHERE node_1 = "Pesticide Residues" RETURN DISTINCT node_1, relation, node_2 LIMIT 1000',
            "metadata": {"type": "cypher", "domain": "who factsheets"},
        },
    ]

    ttl = 2629800  # 1m/30d in seconds
    for data in initial_data_list:
        semantic_cache.store(
            prompt=data["query"],
            response=data["response"],
            metadata=data["metadata"],
            ttl=ttl,
        )

    for data in prebuilt_data_list:
        semantic_cache.store(
            prompt=data["query"],
            response=data["response"],
            metadata=data["metadata"],
            ttl=ttl,
        )

    return semantic_cache


async def init_openai_client() -> AsyncOpenAI:
    client = AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="token-abc123")

    # Warm up local vllm engine.
    await client.chat.completions.create(
        model="yeirr/llama3_2-1B-instruct-awq-g128-4bit",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=False,
        temperature=0.1,
        max_tokens=32,
        stop=["<|eot_id|>", "<|im_end|>", "</s>", "<|end|>"],
    )

    return client


async def init_workflow() -> typing.Any:
    # Initialize workflow.
    workflow = StatefulWorkflow(timeout=60, verbose=False)

    return workflow


async def init_ma_reasoning(
    openai_client: AsyncOpenAI,
    message: str,
    timeout: int,
    cypher_statements_parsed: str,
) -> str:
    ma_messages: typing.List[str] = []

    ddgs_chat_agent_types = [
        "gpt-4o-mini",
        "claude-3-haiku",
        "llama-3.1-70b",  # default
        "mixtral-8x7b",
    ]

    # Identity expert identities.
    classify_expert_identities = await openai_client.chat.completions.create(
        model="yeirr/llama3_2-1B-instruct-awq-g128-4bit",
        messages=[
            {
                "role": "user",
                "content": f"{read_system_templates(system_type='classify_identities')[0] + message}",
            }
        ],
        stream=False,
        temperature=0.1,
        max_tokens=128,
        extra_body={
            "guided_json": expert_identities_schema,
        },
    )

    expert_identities: typing.List[str] = json.loads(
        str(classify_expert_identities.choices[0].message.content)
    )["expert_identities"]

    # Add multi-agent reasoning.
    workers = len(set(expert_identities))

    # Spawn N number of multi-agentic expert responses.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                call_agent_endpoint,
                message,
                ddgs_chat_agent_types[2],
                read_system_templates(system_type=expert_identities[i])[0],
                cypher_statements_parsed,
            )
            for i in range(workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            ma_messages.append(await future.result(timeout=timeout))

    # Convert agents responses to dspy prediction format for aggregation via
    # majority voting(most common response).
    dspy_preds = Completions([{"answer": message} for message in ma_messages])
    # Do not store dynamic reasoning into chat history.
    ma_reasoning = aggregation.majority(dspy_preds)["answer"]

    return str(ma_reasoning)


# Custom events.
class SetUpEvent(Event):
    message: str = Field(description="End user query in natural language.")
    ma_reasoning: str = Field(
        description="Multi-agent reasoning with consensus reached via majority voting."
    )


class LLMEvent(Event):
    payload: str


class StatefulWorkflow(Workflow):
    async_remote_openai_client = AsyncOpenAI(
        base_url=config["BASE_URL"],
        api_key=os.getenv("SNOVA_API_KEY"),
    )

    @step
    async def setup_step(self, ctx: Context, ev: StartEvent) -> SetUpEvent:
        semantic_cache = init_semantic_cache()
        await ctx.set("cache", semantic_cache)

        # Load data into global context.
        await ctx.set("message", ev.message)
        await ctx.set("ma_reasoning", ev.ma_reasoning)
        await ctx.set("memory", ev.memory)
        await ctx.set("cypher_statements_parsed", ev.cypher_statements_parsed)
        await ctx.set("image_buffer", ev.image_buffer)

        template = """[Past Conversations]
        {chat_history}

        [Cypher Statements]
        {kg_response}

        [Reasoning]
        {ma_reasoning}
        
        Given the provided past conversations, Cypher statements and reasoning as context. Answer the following query:
        {query}

        Do not provide preamble, opinions and stick to facts only.
        """
        prompt_template = PromptTemplate(template)
        await ctx.set("prompt_template", prompt_template)

        return SetUpEvent(
            message=ev.message,
            ma_reasoning=ev.ma_reasoning,
            cypher_statements_parsed=ev.cypher_statements_parsed,
        )

    @step
    async def semantic_cache_step(
        self, ctx: Context, ev: SetUpEvent
    ) -> StopEvent | LLMEvent:
        semantic_cache = await ctx.get("cache")
        query = ev.message
        query_vector = semantic_cache._vectorizer.embed(query)

        # Check if response exists in cache.
        result = semantic_cache.check(vector=query_vector)
        if len(result) != 0 and result[0]["metadata"]["type"] == "natural-language":
            # Return as generator for cached response if available and end workflow.
            gen = convert_to_chat_response_generator(result[0]["response"])
            return StopEvent(result=gen)
        else:
            # Else proceed with inference.
            return LLMEvent(payload=query)

    @step
    async def llm_step(self, ctx: Context, ev: LLMEvent) -> StopEvent:
        prompt_template = await ctx.get("prompt_template")
        message = await ctx.get("message")
        ma_reasoning = await ctx.get("ma_reasoning")
        image_buffer = await ctx.get("image_buffer")

        # Streamlit session state for list of 'ChatMessage'.
        memory = await ctx.get("memory")
        kg_response = await ctx.get("cypher_statements_parsed")

        # Parse primitive memory implementation and remove messages with empty content.
        chat_history = [
            {
                "role": "user",
                "content": [{"type": "text", "text": dict(message)["content"]}],
            }
            for message in memory
            if dict(message)["content"] != ""
        ]

        # Truncate excess user-assistant exchanges.
        while len(chat_history) > 10:
            del chat_history[1:2]  # skip system message at index 0

        # Run VLM/LLM inference here.
        if (
            len(image_buffer) != 0
            and isinstance(image_buffer[-1], UploadedFile) is True
        ):
            vlm_timeout = 25
            vlm_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_template.format(
                                chat_history="",
                                kg_response=kg_response,
                                ma_reasoning=ma_reasoning,
                                query=message,
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_buffer[-1].getvalue()).decode('utf-8')}"
                            },
                        },
                    ],
                },
            ]

            try:
                # Default VLM.
                model_id = config["MODEL_ID_VLM_90B"]
                st.session_state["model_metadata"]["model_id"] = model_id
                chat_response = (
                    await self.async_remote_openai_client.chat.completions.create(
                        messages=vlm_messages,
                        timeout=vlm_timeout,
                        temperature=0.1,
                        model=model_id,
                        max_tokens=int(config["MAX_TOKENS"]),
                        stream=True,
                    )
                )
            except openai.RateLimitError as e:
                print(f"{e.status_code} - {e.response}")

                model_id = config["MODEL_ID_VLM_11B"]
                st.session_state["model_metadata"]["model_id"] = model_id
                chat_response = (
                    await self.async_remote_openai_client.chat.completions.create(
                        messages=vlm_messages,
                        timeout=vlm_timeout,
                        temperature=0.1,
                        model=model_id,
                        max_tokens=int(config["MAX_TOKENS"]),
                        stream=True,
                    )
                )
            except openai.APIStatusError as e:
                print(f"{e.status_code} - {e.response}")
                pass
        else:
            llm_timeout = 45
            llm_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_template.format(
                                chat_history=chat_history,
                                kg_response=kg_response,
                                ma_reasoning=ma_reasoning,
                                query=message,
                            ),
                        }
                    ],
                },
            ]

            try:
                # Frontier LLM.
                model_id = config["MODEL_ID_LLM_405B"]
                st.session_state["model_metadata"]["model_id"] = model_id
                chat_response = (
                    await self.async_remote_openai_client.chat.completions.create(
                        messages=llm_messages,
                        timeout=llm_timeout,
                        temperature=0.1,
                        model=model_id,
                        max_tokens=int(config["MAX_TOKENS"]),
                        stream=True,
                    )
                )
            except openai.RateLimitError as e:
                print(f"{e.status_code} - {e.response}")

                model_id = config["MODEL_ID_LLM_70B"]
                st.session_state["model_metadata"]["model_id"] = model_id

                chat_response = (
                    await self.async_remote_openai_client.chat.completions.create(
                        messages=llm_messages,
                        timeout=llm_timeout,
                        temperature=0.1,
                        model=model_id,
                        max_tokens=int(config["MAX_TOKENS"]),
                        stream=True,
                    )
                )
            except openai.APIStatusError as e:
                print(f"{e.status_code} - {e.response}")
                pass

        # Return an async generator.
        return StopEvent(result=chat_response)


# Schemas.
expert_identities_enum = [
    "biology",
    "calculus",
    "chemistry",
    "generic",
    "macroeconomics",
    "medicine",
    "microeconomics",
    "philosophy",
    "physics",
    "probability",
    "diet",
    "foodscience",
    "nutrition",
]
expert_identities_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "deprecated": False,
    "readOnly": True,
    "writeOnly": False,
    "title": "Expert identities",
    "required": ["expert_identities"],
    "type": "object",
    "properties": {
        "expert_identities": {
            "description": "Generate multiple expert identities for multi-agents reasoning.",
            "type": "array",
            "uniqueItems": True,
            "minItems": 1,
            "maxItems": 5,
            "unevaluatedItems": False,
            "items": {"type": "string", "enum": expert_identities_enum},
        }
    },
}


async def main(timeout: int = 60) -> None:
    semantic_cache = init_semantic_cache()
    openai_client = await init_openai_client()
    workflow = await init_workflow()

    # Set a default model.
    if "model_metadata" not in st.session_state:
        st.session_state["model_metadata"] = {
            "model_type": "llm",
            "model_id": config["MODEL_ID_LLM_8B"],
            "rate_limited": False,
        }

    # Initialize chat history.
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=config["SYSTEM_MESSAGE"],
            )
        )

    # Initialize image support.
    if "render_image" not in st.session_state:
        st.session_state["render_image"] = False

    if "image_buffer" not in st.session_state:
        st.session_state["image_buffer"] = []

    # Display chat messages from history on app rerun.
    for message in st.session_state.messages:
        if dict(message)["role"] == "system" or dict(message)["content"] == "":
            st.empty()
        else:
            with st.chat_message(dict(message)["role"]):
                st.markdown(dict(message)["content"])

    # Placeholders.
    text_buffer: typing.List[str] = []
    cypher_statements: typing.List[str] = []

    with bottom():
        # Accept user text and image input.
        with st.container(height=None, border=None):
            # Load file image from host file system into memory.
            uploaded_file = st.file_uploader(
                "Select file",
                accept_multiple_files=False,
            )

            if uploaded_file is not None:
                st.session_state["render_image"] = True
                if isinstance(uploaded_file, UploadedFile):
                    st.session_state["image_buffer"].append(uploaded_file)

            else:
                st.session_state["render_image"] = False

            # Display image if available.
            if uploaded_file is not None and st.session_state["render_image"] is True:
                st.image(uploaded_file, width=64)

            # Text input.
            message = st.chat_input("Message")

            # Pin important message to bottom.
            st.write(
                "Language models can hallucinate, please verify output with trusted sources. Rate limits apply."
            )

    if message:
        # Vectorize query.
        query_vector = semantic_cache._vectorizer.embed(message)

        # Add user message to chat history.
        st.session_state.messages.append(
            ChatMessage(role=MessageRole.USER, content=message)
        )

        # Display user message in chat message container.
        with st.chat_message("user"):
            st.markdown(dict(st.session_state.messages[-1])["content"])

        # Display assistant response in chat message container.
        with st.chat_message("assistant"):
            with st.status("Parsing knowledge graph..."):
                # Render kg triples here.
                result = semantic_cache.check(vector=query_vector)
                if len(result) != 0 and result[0]["metadata"]["type"] == "cypher":
                    prebuilt_cypher_query = result[0]["response"]
                    cypher_statements.append(
                        await run_cypher_query(cypher_query=prebuilt_cypher_query)
                    )
                    st.markdown(f"{cypher_statements[0]}")
                else:
                    cypher_statements.append("No cypher statements.")
                    st.write("CYPHER CACHE MISS")

            with st.status("Reasoning..."):
                ma_reasoning = await init_ma_reasoning(
                    openai_client=openai_client,
                    message=message,
                    timeout=timeout,
                    cypher_statements_parsed=cypher_statements[0],
                )
                st.write(ma_reasoning)

            # Use llama-index messages format and custom defined workflow.
            handler = workflow.run(
                message=message,
                ma_reasoning=ma_reasoning,
                memory=st.session_state.messages,
                cypher_statements_parsed=cypher_statements[0],
                image_buffer=st.session_state["image_buffer"]
                if st.session_state["image_buffer"] is not None
                else None,
            )

            # Typewriter effect: replace each displayed chunk.
            with st.empty():
                async for ev in handler.stream_events():
                    if isinstance(ev, StopEvent):
                        async for chunk in ev.result:
                            if chunk.choices[0].finish_reason != "stop":
                                if chunk.choices[0].delta.role == "assistant":
                                    text_buffer.append(chunk.choices[0].delta.content)
                                st.write("".join(text_buffer))

            # Write unique responses to semantic cache.
            result = semantic_cache.check(vector=query_vector)
            if len(result) == 0:
                semantic_cache.store(
                    message,
                    "".join(text_buffer),
                    query_vector,
                    metadata={
                        "type": "natural-language",
                        "model": st.session_state["model_metadata"]["model_id"],
                        "uid": "testuser",
                        "timestamp": datetime.now(timezone.utc).timestamp(),
                    },
                )

    # Write buffered response to history and current session.
    assistant_response = "".join(text_buffer)
    st.session_state.messages.append(
        ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response)
    )

    # Clear image buffer each request.
    st.session_state["image_buffer"] = []


if __name__ == "__main__":
    asyncio.run(main())
