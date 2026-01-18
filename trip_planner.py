from dotenv import load_dotenv
import os
import streamlit as st
import asyncio
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from pydantic import BaseModel

set_tracing_disabled(disabled=True)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

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
        model="gemini-2.5-flash-lite",
        openai_client=client
    ),
    instructions=(
        "Create a day-by-day travel itinerary. Be concise and respond in one message. "
        "Include key attractions and activities for each day."
    ),
)

# ---- Budget Agent (estimates costs under constraints) ----
budget_agent = Agent(
    name="Budget Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash-lite",
        openai_client=client
    ),
    instructions=(
        "Estimate travel costs for lodging, food, transport, and activities. Be concise and respond in one message. "
        "Provide a breakdown and total estimate."
    ),
)

# ---- Local Guide Agent (adds local tips & dining) ----
local_guide_agent = Agent(
    name="Local Guide Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash-lite",
        openai_client=client
    ),
    instructions=(
        "Provide local food recommendations, restaurant suggestions, and cultural tips. Be concise and respond in one message."
    ),
)

# ---- Core orchestrator: Travel Agent ----
travel_agent = Agent(
    name="Travel Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash-lite",
        openai_client=client
    ),
    instructions=(
        "You orchestrate travel planning by calling these agents IN PARALLEL (all at once, not sequentially):\n"
        "1. planner_agent - for the itinerary\n"
        "2. budget_agent - for cost estimates\n"
        "3. local_guide_agent - for food and local tips\n\n"
        "IMPORTANT: Call all three agents in a SINGLE turn to minimize API calls.\n"
        "After receiving their responses, combine into the final JSON output immediately.\n"
        "Do NOT use tavily_search - the agents have the knowledge needed."
    ),
    output_type=TravelOutput,
    tools=[
        planner_agent.as_tool(
            tool_name="planner_agent",
            tool_description="Creates day-by-day itinerary"),
        budget_agent.as_tool(
            tool_name="budget_agent",
            tool_description="Estimates trip costs"),
        local_guide_agent.as_tool(
            tool_name="local_guide_agent",
            tool_description="Provides food and local tips")
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
    result = await Runner.run(travel_agent, prompt, max_turns=10)
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