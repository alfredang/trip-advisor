from dotenv import load_dotenv
import os
import streamlit as st
import asyncio
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
from pydantic import BaseModel
from tavily import TavilyClient

set_tracing_disabled(disabled=True)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")

@function_tool
def tavily_search(query: str) -> str:
    """Search the internet using Tavily API for current information."""
    tavily = TavilyClient(api_key=tavily_key)
    response = tavily.search(query=query, search_depth="basic")

    results = response.get('results', [])
    summary = "\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results])
    return summary

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Streamlit Page Config ----
st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="🧳",
    layout="wide"
)

st.title("🧳 AI Trip Planner (Multi-Agent System)")
st.markdown("Plan your perfect trip with our team of specialized AI agents!")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class TravelOutput(BaseModel):
    destination: str
    duration: str
    summary: str
    cost: str
    tips: str

# ---- Planner Agent (builds day-by-day itinerary) ----
planner_agent = Agent(
    name="Planner Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client
    ),
    handoff_description="Use me when the user asks to plan or outline an itinerary, schedule, or daily plan.",
    instructions=(
        "You specialize in building day-by-day travel itineraries and sequencing activities. "
        'Always return JSON with this structure: {"destination":"string","duration":"string","summary":"string"}.'
    ),
    tools=[
        tavily_search,
    ]
)

# ---- Budget Agent (estimates costs under constraints) ----
budget_agent = Agent(
    name="Budget Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client
    ),
    handoff_description="Use me when the user mentions budget, price, cost, dollars, under $X, or asks 'how much'.",
    instructions=(
        "You estimate costs for lodging, food, transport, and activities at a high level; flag budget violations. "
        'Always return JSON with this structure: {"cost":"string"}.'
    ),
)

# ---- Local Guide Agent (adds local tips & dining) ----
local_guide_agent = Agent(
    name="Local Guide Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client
    ),
    handoff_description="Use me when the user asks for food, restaurants, neighborhoods, local tips, or 'what's good nearby'.",
    instructions=(
        "You provide restaurants, neighborhoods, cultural tips, and current local highlights. "
        'Always return JSON with this structure: {"tips":"string"}.'
    ),
    tools=[
        tavily_search,
    ]
)

# ---- Core orchestrator: Travel Agent ----
travel_agent = Agent(
    name="Travel Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client
    ),
    instructions=(
        "You are a friendly and knowledgeable travel planner that helps users plan trips, suggest destinations, and create detailed summaries of their journeys.\n"
        "Your primary role is to orchestrate other specialized agents (used as tools) to complete the user's request.\n"
        "\n"
        "When planning an itinerary, call the **Planner Agent** to create daily schedules, organize destinations, and recommend attractions or activities. Do not create itineraries yourself.\n"
        "When estimating costs, call the **Budget Agent** to calculate the total trip cost including flights, hotels, and activities. Do not calculate or estimate prices on your own.\n"
        "When recommending local experiences, restaurants, neighborhoods, or cultural highlights, call the **Local Guide Agent** to provide these insights. Do not generate local recommendations without this agent.\n"
        "\n"
        "Use these agents one at a time in a logical order based on the request — start with the Planner Agent, then the Budget Agent, and finally the Local Guide Agent.\n"
        "After receiving results from these agents, combine their outputs into a single structured summary.\n"
        "\n"
        "Return JSON output using this exact structure:\n"
        "{\"destination\": \"string\", \"duration\": \"string\", \"summary\": \"string\", \"cost\": \"string\", \"tips\": \"string\"}.\n"
    ),
    output_type=TravelOutput,
    tools=[
        tavily_search,
        planner_agent.as_tool(
            tool_name="planner_agent",
            tool_description="plan or outline an itinerary, schedule, or daily plan"),
        budget_agent.as_tool(
            tool_name="budget_agent",
            tool_description="calculates the cost of a trip"),
        local_guide_agent.as_tool(
            tool_name="local_guide_agent",
            tool_description="provide restaurants, neighborhoods, cultural tips, and current local highlights")
    ]
)

# ---- Sidebar: User Inputs ----
st.sidebar.header("📝 Trip Details")

destination = st.sidebar.text_input(
    "🌍 Destination",
    value="Tokyo",
    help="Enter your travel destination"
)

num_days = st.sidebar.number_input(
    "📅 Number of Days",
    min_value=1,
    max_value=30,
    value=5,
    help="How many days is your trip?"
)

budget = st.sidebar.number_input(
    "💵 Budget (USD)",
    min_value=100,
    max_value=50000,
    value=2000,
    step=100,
    help="Your total trip budget in US dollars"
)

preferences = st.sidebar.text_area(
    "✨ Special Preferences (optional)",
    placeholder="e.g., vegetarian food, museums, adventure activities, family-friendly...",
    help="Any special requirements or interests for your trip"
)

# ---- Agent Info Display ----
with st.sidebar.expander("🤖 Meet the AI Agents"):
    st.markdown("""
    **🧠 Planner Agent**
    Creates your day-by-day itinerary

    **💰 Budget Agent**
    Estimates and tracks all costs

    **🍣 Local Guide Agent**
    Recommends food & local tips

    **✈️ Travel Agent**
    Orchestrates all agents for the final plan
    """)

# ---- Async Runner Function ----
async def generate_trip_plan(destination: str, num_days: int, budget: int, preferences: str) -> TravelOutput:
    """Run the multi-agent system to generate a trip plan."""
    prompt = f"""Plan a {num_days}-day trip to {destination} with a budget of ${budget} USD.

{'Special preferences: ' + preferences if preferences else 'No special preferences.'}

Please create a comprehensive travel plan including:
1. A detailed day-by-day itinerary
2. Estimated costs for accommodations, food, transportation, and activities
3. Local food recommendations and cultural tips
"""
    result = await Runner.run(travel_agent, prompt, max_turns=30)
    return result.final_output

# ---- Main App Logic ----
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    plan_button = st.button("🚀 Generate Trip Plan", type="primary", use_container_width=True)

if plan_button:
    with st.spinner("🤖 AI agents are working on your trip plan..."):
        # Status placeholders for each agent
        status_container = st.container()
        with status_container:
            st.info("✈️ **Travel Agent** is coordinating the team...")
            planner_status = st.empty()
            budget_status = st.empty()
            guide_status = st.empty()

            planner_status.info("🧠 **Planner Agent** is creating your itinerary...")
            budget_status.info("💰 **Budget Agent** is calculating costs...")
            guide_status.info("🍣 **Local Guide Agent** is finding local gems...")

        try:
            # Run the async function
            result = asyncio.run(generate_trip_plan(destination, num_days, budget, preferences))

            # Clear status messages
            status_container.empty()

            st.success("✅ Trip plan generated successfully!")
            st.divider()

            # Display Results
            st.header(f"🗺️ Your {result.duration} Trip to {result.destination}")

            # Create tabs for different sections
            tab1, tab2, tab3 = st.tabs(["📋 Itinerary", "💰 Budget Breakdown", "🍣 Local Tips"])

            with tab1:
                st.subheader("🧠 Day-by-Day Itinerary")
                st.markdown(result.summary)

            with tab2:
                st.subheader("💰 Cost Estimates")
                st.markdown(result.cost)

                # Budget comparison
                st.divider()
                st.metric(
                    label="Your Budget",
                    value=f"${budget:,}",
                    delta=None
                )

            with tab3:
                st.subheader("🍣 Local Recommendations & Tips")
                st.markdown(result.tips)

            # Download option
            st.divider()
            full_plan = f"""# {result.destination} Trip Plan ({result.duration})

## Itinerary
{result.summary}

## Budget
{result.cost}

## Local Tips
{result.tips}
"""
            st.download_button(
                label="📥 Download Trip Plan",
                data=full_plan,
                file_name=f"{destination.lower().replace(' ', '_')}_trip_plan.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please check your API key and try again.")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>Powered by Multi-Agent AI System</div>",
    unsafe_allow_html=True
)